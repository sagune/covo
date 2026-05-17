import re
import torch
import pytorch_lightning as pl
from .utils import KWSOutput
from .resnet import Resnet
from .discriminator import Discriminator
from .heads import ResNetDiscriminator, ResNetDiscriminatorLarge
from .entropyLoss import HLoss
from .dannce import DANNCE
from torch.optim import Adam
from torchmetrics import PrecisionRecallCurve, Accuracy
import numpy as np
import pandas as pd
from confidence_intervals import evaluate_with_conf_int


class KWSModel(pl.LightningModule):
    def __init__(
        self,
        backbone: str = 'resnet',
        tcresnet_channels: int = 64,
        tcresnet_blocks: int = 6,
        tcresnet_dropout: float = 0.1,
        subsequence_aux: bool = False,
        subsequence_aux_weight: float = 0.2,
        subsequence_aux_temperature: float = 0.5,
        decision_threshold_mode: str = 'max_fbeta',
        decision_threshold_fixed: float = 0.5,
        decision_threshold_override: float = None,
        decision_threshold_fbeta: float = 0.5,
        decision_threshold_min_recall: float = 0.0,
        decision_threshold_min_precision: float = 0.0,
        decision_threshold_grid_size: int = 201,
        large_heads: bool = False,
        adversarial_training: bool = False,
        dannce: bool = False,
        adversarial_examples_ratio: float = 0.5,
        adversarial_examples_lr: float = 1.5e-6,
        adversarial_train_steps: int = 5,
        adv_kl_weight: float = 1.0,
        entropy: bool = False,
        domain_adversary_weight: float = 0.1,
        entropy_weight: float = 0.1,
        supression_decay: float = 1e-3,
        early_adversary_supression: bool = True,
        num_domains: int = 72,
        sampling: str = 'utterance-examples',
        resample_every_epoch: bool = True,
        kw_type: str = 'tts',
        kw_p: float = 0.5,
        batch_size: int = 1,
        accumulate_grad_batches: int = 1,
        learning_rate: float = 1e-4,
        features_lr: float = 1e-4,
        classifier_lr: float = 1e-4,
        discriminator_lr: float = 1e-4,
        lr_step: int = 40,
        weight_decay: float = 0.,
        beta_1: float = 0.9,
        beta_2: float = 0.99
    ):
        super().__init__()

        # save hyperparameters
        self.save_hyperparameters()
        # backward compatibility: older checkpoints may miss some hparams
        if not hasattr(self.hparams, 'adversarial_training'):
            self.hparams.adversarial_training = adversarial_training
        if not hasattr(self.hparams, 'entropy'):
            self.hparams.entropy = entropy
        if not hasattr(self.hparams, 'num_domains'):
            self.hparams.num_domains = num_domains
        if not hasattr(self.hparams, 'backbone'):
            self.hparams.backbone = backbone
        if not hasattr(self.hparams, 'tcresnet_channels'):
            self.hparams.tcresnet_channels = tcresnet_channels
        if not hasattr(self.hparams, 'tcresnet_blocks'):
            self.hparams.tcresnet_blocks = tcresnet_blocks
        if not hasattr(self.hparams, 'tcresnet_dropout'):
            self.hparams.tcresnet_dropout = tcresnet_dropout
        if not hasattr(self.hparams, 'subsequence_aux'):
            self.hparams.subsequence_aux = subsequence_aux
        if not hasattr(self.hparams, 'subsequence_aux_weight'):
            self.hparams.subsequence_aux_weight = subsequence_aux_weight
        if not hasattr(self.hparams, 'subsequence_aux_temperature'):
            self.hparams.subsequence_aux_temperature = subsequence_aux_temperature
        if not hasattr(self.hparams, 'decision_threshold_mode'):
            self.hparams.decision_threshold_mode = decision_threshold_mode
        if not hasattr(self.hparams, 'decision_threshold_fixed'):
            self.hparams.decision_threshold_fixed = decision_threshold_fixed
        if not hasattr(self.hparams, 'decision_threshold_override'):
            self.hparams.decision_threshold_override = decision_threshold_override
        if not hasattr(self.hparams, 'decision_threshold_fbeta'):
            self.hparams.decision_threshold_fbeta = decision_threshold_fbeta
        if not hasattr(self.hparams, 'decision_threshold_min_recall'):
            self.hparams.decision_threshold_min_recall = decision_threshold_min_recall
        if not hasattr(self.hparams, 'decision_threshold_min_precision'):
            self.hparams.decision_threshold_min_precision = decision_threshold_min_precision
        if not hasattr(self.hparams, 'decision_threshold_grid_size'):
            self.hparams.decision_threshold_grid_size = decision_threshold_grid_size

        # instantiate the model
        # it contains the feature extractor and the classifier
        self.model = Resnet(
           num_channels=1,
           num_classes=2,
           backbone=self.hparams.backbone,
           tcresnet_channels=self.hparams.tcresnet_channels,
           tcresnet_blocks=self.hparams.tcresnet_blocks,
           tcresnet_dropout=self.hparams.tcresnet_dropout,
        )
        self.subsequence_aux = bool(self.hparams.subsequence_aux)
        if self.subsequence_aux:
            self.subseq_head = torch.nn.Sequential(
                torch.nn.Conv1d(1, 16, kernel_size=7, padding=3, bias=False),
                torch.nn.BatchNorm1d(16),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv1d(16, 1, kernel_size=1, bias=True),
            )

        if self.hparams.adversarial_training:
            # disable automatic optimization
            self.automatic_optimization = False
            # instantiate the discriminator
            discriminator_head = ResNetDiscriminator if not self.hparams.large_heads else ResNetDiscriminatorLarge
            self.discriminator = Discriminator(
                head = discriminator_head(
                    in_features = self.model.feature_dim,
                    num_labels = self.hparams.num_domains
                ),
                reverse = True
            )
            # instantiate the Accuracy module
            self.accuracy = Accuracy(task='multiclass', num_classes=self.hparams.num_domains)

        # instantiate a PrecisionRecallCurve object
        self.pr_curve = PrecisionRecallCurve(task='binary')
        self._eval_threshold = float(self.hparams.decision_threshold_fixed)

    @staticmethod
    def _safe_div(num: float, den: float) -> float:
        return float(num) / float(den) if float(den) > 0.0 else 0.0

    def _metrics_at_threshold(self, labels: torch.Tensor, samples: torch.Tensor, threshold: float):
        preds = (samples >= float(threshold)).long()
        labels = labels.long()
        tp = int(((preds == 1) & (labels == 1)).sum().item())
        fp = int(((preds == 1) & (labels == 0)).sum().item())
        fn = int(((preds == 0) & (labels == 1)).sum().item())
        precision = self._safe_div(tp, tp + fp)
        recall = self._safe_div(tp, tp + fn)
        return precision, recall

    def _f_beta(self, precision: float, recall: float, beta: float) -> float:
        b2 = float(beta) ** 2
        den = b2 * precision + recall
        return (1.0 + b2) * precision * recall / den if den > 0.0 else 0.0

    def _select_threshold(self, labels: torch.Tensor, samples: torch.Tensor) -> float:
        mode = str(self.hparams.decision_threshold_mode).lower()
        if mode == 'fixed':
            return float(self.hparams.decision_threshold_fixed)

        grid_size = max(11, int(self.hparams.decision_threshold_grid_size))
        # Evaluate on a dense grid for stability.
        cand = torch.linspace(0.0, 1.0, steps=grid_size, device=samples.device).tolist()
        min_recall = float(self.hparams.decision_threshold_min_recall)
        min_precision = float(self.hparams.decision_threshold_min_precision)
        beta = float(self.hparams.decision_threshold_fbeta)

        best_thr = float(self.hparams.decision_threshold_fixed)
        best_score = -1.0
        best_prec = -1.0
        best_rec = -1.0

        for thr in cand:
            precision, recall = self._metrics_at_threshold(labels=labels, samples=samples, threshold=thr)
            if recall < min_recall or precision < min_precision:
                continue
            if mode == 'max_precision':
                score = precision
            elif mode == 'max_precision_at_min_recall':
                score = precision
            else:
                # default: precision-biased F-beta (beta<1 favors precision).
                score = self._f_beta(precision, recall, beta=beta)

            # tie-break toward higher precision, then recall.
            if (score > best_score) or (score == best_score and precision > best_prec) or (score == best_score and precision == best_prec and recall > best_rec):
                best_score = score
                best_thr = float(thr)
                best_prec = precision
                best_rec = recall

        return best_thr

    def forward(
        self,
        input_features: torch.Tensor,
        labels: torch.Tensor = None
    ):
        features, feature_map = self.model.extract_features(input_features)
        logits = self.model.classifier(features)
        aux_loss = None
        if labels != None:
            loss = torch.nn.functional.cross_entropy(logits, labels.view(-1))
            if self.subsequence_aux:
                # Weakly-supervised subsequence matching loss:
                # encourage at least one strong local match on positives.
                temporal_map = feature_map.mean(dim=1).mean(dim=1)  # [B, T]
                temporal_logits = self.subseq_head(temporal_map.unsqueeze(1)).squeeze(1)  # [B, T]
                temp = max(1e-3, float(self.hparams.subsequence_aux_temperature))
                pooled_logits = torch.logsumexp(temporal_logits / temp, dim=-1) * temp
                aux_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    pooled_logits,
                    labels.view(-1).float(),
                )
                loss = loss + float(self.hparams.subsequence_aux_weight) * aux_loss
        else:
            loss = None
        return KWSOutput(
            loss=loss,
            logits=logits,
            features=features,
            aux_loss=aux_loss,
        )
    
    def on_train_epoch_start(self):
        # calculate suppression factor of the adversarial loss
        # and apply it if needed
        if self.hparams.adversarial_training or self.hparams.entropy:
            self.supression = (2.0 / (1. + np.exp(-self.hparams.supression_decay * self.trainer.current_epoch)) - 1)
            print(f'supression={self.supression:.2f}')

        if self.hparams.adversarial_training:
            self.beta = 1 * self.hparams.domain_adversary_weight
            if self.hparams.early_adversary_supression:
                self.beta *= self.supression
            print(f'beta={self.beta:.2f}')
            self.discriminator.set_beta(self.beta)

    def training_step(self, batch, batch_idx): 

        # get optimizers and lr_schedulers if in adversarial training mode
        # and initialize running losses
        if self.hparams.adversarial_training:
            f_opt, c_opt, d_opt = self.optimizers()
            f_sch, c_sch, d_sch = self.lr_schedulers()
            running_c_loss, running_d_loss, running_e_loss = (0., 0., 0.)
        else:
            running_c_loss = 0.
        running_aux_loss = 0.

        if self.hparams.kw_type == 'all':
            # whether to use tts or natural -based keywords
            k_mask = torch.rand(int(batch['features'].size(dim=0) / 2)) > self.hparams.kw_p
            k_mask = torch.stack((k_mask, torch.logical_not(k_mask)), dim=1).flatten()
            input_features = batch['features'][k_mask]
            c_labels = batch['labels'][k_mask]
            if self.hparams.adversarial_training:
                d_labels = batch['domain'][k_mask]
        else:
            input_features = batch['features']
            c_labels = batch['labels']
            if self.hparams.adversarial_training:
                d_labels = batch['domain']
        batch_size = input_features.size(dim=0)

        # get DANNCE new examples
        if self.hparams.dannce and self.hparams.adversarial_training:
            minibatch_size = batch_size // self.hparams.accumulate_grad_batches
            minibatch_indices = list(zip(range(0, batch_size, minibatch_size), range(minibatch_size, batch_size + minibatch_size, minibatch_size)))
            for idx0, idx1 in minibatch_indices:
                input_features[idx0:idx1] = DANNCE.train_adversarial_examples(
                    input_features = input_features[idx0:idx1],
                    d_labels = d_labels[idx0:idx1],
                    adversarial_examples_ratio = self.hparams.adversarial_examples_ratio,
                    adversarial_examples_lr = self.hparams.adversarial_examples_lr,
                    adversarial_train_steps = self.hparams.adversarial_train_steps,
                    adv_kl_weight = self.hparams.adv_kl_weight,
                    model = self,
                    domain_adversary = self.discriminator,
                    domain_adversary_weight = self.hparams.domain_adversary_weight
                    #logger = ...,
                )

        # manually reset gradients when in adversarial training mode
        if self.hparams.adversarial_training:
            f_opt.zero_grad()
            c_opt.zero_grad()
            d_opt.zero_grad()

        # loop through minibatches
        minibatch_size = batch_size // self.hparams.accumulate_grad_batches
        minibatch_indices = [(0, batch_size)] if not self.hparams.adversarial_training else list(zip(range(0, batch_size, minibatch_size), range(minibatch_size, batch_size + minibatch_size, minibatch_size)))
        num_minibatches = len(minibatch_indices)
        d_preds = []
        for idx0, idx1 in minibatch_indices:

            # forward minibatch through the model
            kws_output = self.forward(
                input_features = input_features[idx0:idx1],
                labels = c_labels[idx0:idx1]
            )

            # forward features and domain labels through the discriminator
            if self.hparams.adversarial_training:
                adv_output = self.discriminator(
                    input_features = kws_output.features,
                    labels = d_labels[idx0:idx1], 
                    use_grad_reverse = True
                )        
                d_preds.append(torch.argmax(adv_output.logits, dim=-1))

            # compute training loss(es)
            c_loss = kws_output.loss
            loss = c_loss.clone()
            if self.hparams.adversarial_training:
                d_loss = adv_output.loss
                loss += d_loss
            if self.hparams.entropy:
                entropy_loss = HLoss()
                e_loss = entropy_loss(kws_output.logits)
                if self.hparams.early_adversary_supression:
                    e_loss = e_loss * (self.supression * self.hparams.entropy_weight)
                loss += e_loss

            # manual backpropagation in adversarial training mode
            if self.hparams.adversarial_training:
                self.manual_backward(loss)

            # accumulate running losses for logging
            running_c_loss += c_loss.detach() / num_minibatches
            if self.hparams.adversarial_training:
                running_d_loss += d_loss.detach() / num_minibatches
            if self.hparams.entropy:
                running_e_loss += e_loss.detach() / num_minibatches
            if kws_output.aux_loss is not None:
                running_aux_loss += kws_output.aux_loss.detach() / num_minibatches

        # log losses
        self.log('train/class_loss', running_c_loss, batch_size=batch_size, sync_dist=True)
        if self.hparams.adversarial_training:
            self.log('train/domain_loss', running_d_loss, batch_size=batch_size, sync_dist=True)
            # compute discriminator accuracy and log
            d_accuracy = self.accuracy(torch.cat(d_preds, dim=0), d_labels)
            self.log('train/discriminator_acc', d_accuracy.detach(), batch_size=batch_size, sync_dist=True)
        if self.hparams.entropy:
            self.log('train/entropy_loss', running_e_loss, batch_size=batch_size, sync_dist=True)
        if self.subsequence_aux:
            self.log('train/subsequence_aux_loss', running_aux_loss, batch_size=batch_size, sync_dist=True)

        # manually step optimizers in adversarial training mode
        if self.hparams.adversarial_training:
            f_opt.step()
            c_opt.step()
            d_opt.step()
            # step the learning rate schedulers
            if self.trainer.is_last_batch:
                f_sch.step()
                c_sch.step()
                d_sch.step()

        # return final loss
        # important when not in adversarial training mode
        return loss

    def on_validation_epoch_start(self):
        # list for storing the outputs of each validation step
        # in the past this was done automatically using the validation_epoch_end hook
        # but https://github.com/Lightning-AI/lightning/pull/16520
        self.validation_step_outputs = []

    def validation_step(self, batch, batch_idx, dataloader_idx=0):        
        # forward each group through the model
        output = []
        for input_features, labels in zip(batch['features'], batch['hotword_labels']):
            output.append(self.forward(
                input_features = input_features,
                labels = labels
            ))
        batch_size = batch['features'][0].size(dim=0)
        # get loss
        loss = sum(out.loss for out in output).detach()

        if batch.get('hotword_mask', None) != None:
            preds = torch.cat([out.logits.softmax(dim=-1)[:, 1].detach() * mask for out, mask in zip(output, batch['hotword_mask'])], dim=0)
        else:
            preds = torch.cat([out.logits.softmax(dim=-1)[:, 1].detach() for out in output], dim=0)
        targets = torch.cat([labels for labels in batch['hotword_labels']], dim=0)

        # append to validation step outputs for correct dataloader
        if dataloader_idx > len(self.validation_step_outputs) - 1:
            self.validation_step_outputs += [[]] * (dataloader_idx - len(self.validation_step_outputs) + 1)
        self.validation_step_outputs[dataloader_idx].append({
            'loss': loss,
            'preds': preds,
            'targets': targets
        })

    def on_validation_epoch_end(self):

        # to store the average metrics
        avg_metrics = {metric: 0. for metric in ['val/loss', 'metrics/precision', 'metrics/recall', 'metrics/f1']}
        zh_metrics = {metric: 0. for metric in ['val/loss_zh', 'metrics/precision_zh', 'metrics/recall_zh', 'metrics/f1_zh']}
        en_metrics = {metric: 0. for metric in ['val/loss_en', 'metrics/precision_en', 'metrics/recall_en', 'metrics/f1_en']}
        zh_preds_all, zh_targets_all = [], []

        # loop through the results of each dataloader
        for dataloader_idx, validation_step_outputs in enumerate(self.validation_step_outputs):
            preds_cat = torch.cat([step_output['preds'] for step_output in validation_step_outputs], dim=0)
            targets_cat = torch.cat([step_output['targets'] for step_output in validation_step_outputs], dim=0)
            threshold = self._select_threshold(labels=targets_cat, samples=preds_cat)
            precision_v, recall_v = self._metrics_at_threshold(labels=targets_cat, samples=preds_cat, threshold=threshold)
            f1_v = self._f_beta(precision_v, recall_v, beta=1.0)
            metrics = {
                'val/loss' + '_' + str(dataloader_idx) : sum(step_output['loss'] for step_output in validation_step_outputs) / len(validation_step_outputs),
                'metrics/precision' + '_' + str(dataloader_idx) : precision_v,
                'metrics/recall' + '_' + str(dataloader_idx) : recall_v,
                'metrics/f1' + '_' + str(dataloader_idx) : f1_v,
                'metrics/threshold' + '_' + str(dataloader_idx) : threshold,
            }

            # add to the average metrics
            for key in avg_metrics.keys():
                avg_metrics[key] += metrics[key + '_' + str(dataloader_idx)] / len(self.validation_step_outputs)

            if dataloader_idx == 0 or dataloader_idx == 1:
                zh_preds_all.append(preds_cat)
                zh_targets_all.append(targets_cat)
                for key in avg_metrics.keys():
                    zh_metrics[key + '_zh'] += metrics[key + '_' + str(dataloader_idx)] / 2
            elif dataloader_idx == 2 or dataloader_idx == 3:
                for key in avg_metrics.keys():
                    en_metrics[key + '_en'] += metrics[key + '_' + str(dataloader_idx)] / 2

            # log metrics that can be averaged across processes
            self.log_dict(metrics, sync_dist=True) 

        # log average metrics
        self.log_dict(avg_metrics, sync_dist=True)
        self.log_dict(zh_metrics, sync_dist=True)
        self.log_dict(en_metrics, sync_dist=True)
        if len(zh_preds_all) > 0:
            zh_preds = torch.cat(zh_preds_all, dim=0)
            zh_targets = torch.cat(zh_targets_all, dim=0)
            self._eval_threshold = self._select_threshold(labels=zh_targets, samples=zh_preds)
            self.log('metrics/decision_threshold_zh', self._eval_threshold, sync_dist=True)

        # empty validation step outputs list
        self.validation_step_outputs = []

    def configure_optimizers(self):
        if not self.hparams.adversarial_training:
            optimizer = Adam(
                self.parameters(),
                lr = self.hparams.learning_rate,
                betas = (self.hparams.beta_1, self.hparams.beta_2),
                weight_decay = self.hparams.weight_decay
            )
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size = self.hparams.lr_step,
                gamma = 1e-1
            ) 
            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
        else:
            classifier_module = self.model.classifier if not self.subsequence_aux else torch.nn.ModuleList([self.model.classifier, self.subseq_head])
            optimizers = [
                Adam(
                    nn_.parameters(),
                    lr = lr_,
                    betas = (self.hparams.beta_1, self.hparams.beta_2),
                    weight_decay = self.hparams.weight_decay
                )
            for nn_, lr_ in zip([self.model.feature_extractor, classifier_module, self.discriminator], [self.hparams.features_lr, self.hparams.classifier_lr, self.hparams.discriminator_lr])]
            schedulers = [
                torch.optim.lr_scheduler.StepLR(
                    optim,
                    step_size = self.hparams.lr_step,
                    gamma = 1e-1
                ) 
            for optim in optimizers]
            return (optimizers, schedulers)

    def on_test_epoch_start(self):
        # list for storing the outputs of each test step
        # in the past this was done automatically using the test_epoch_end hook
        # but https://github.com/Lightning-AI/lightning/pull/16520
        self.test_step_outputs = []

    def test_step(self, batch, batch_idx):     

        # forward each group through the model
        output = []
        output3 = []
        for features, labels in zip(batch['features'], batch['hotword_labels']):
            output.append(self.forward(
                input_features = features,
                labels = labels
            ))

        # join predicted class probabilities and target class indices
        if batch.get('hotword_mask', None) != None:
            preds = torch.cat([out.logits.softmax(dim=-1)[:, 1].detach() * mask for out, mask in zip(output, batch['hotword_mask'])], dim=0)
        else:
            preds = torch.cat([out.logits.softmax(dim=-1)[:, 1].detach() for out in output], dim=0)

        targets = torch.cat([labels for labels in batch['hotword_labels']], dim=0)
        self.test_step_outputs.append({
            'preds': preds,
            'targets': targets,
            'speaker': batch['speaker']
        })

    def on_test_epoch_end(self):
        threshold_override = getattr(self.hparams, 'decision_threshold_override', None)
        threshold = (
            float(threshold_override)
            if threshold_override is not None
            else float(getattr(self, '_eval_threshold', self.hparams.decision_threshold_fixed))
        )

        def f_precision(labels, samples, samples2=None):
            precision, _ = self._metrics_at_threshold(labels=labels, samples=samples, threshold=threshold)
            return precision
        
        def f_recall(labels, samples, samples2=None):
            _, recall = self._metrics_at_threshold(labels=labels, samples=samples, threshold=threshold)
            return recall
        
        def f_f1(labels, samples, samples2=None):
            precision, recall = self._metrics_at_threshold(labels=labels, samples=samples, threshold=threshold)
            return self._f_beta(precision, recall, beta=1.0)
        
        # set conditions based on speaker
        speakers = [step_output['speaker'] for step_output in self.test_step_outputs for i_ in range(len(step_output['preds']))]
        speaker2id = {speaker: speaker_id for speaker_id, speaker in enumerate(set(speakers))}
        conditions = [speaker2id[speaker] for speaker in speakers]

        samples = torch.cat([step_output['preds'] for step_output in self.test_step_outputs], dim=0)
        labels = torch.cat([step_output['targets'] for step_output in self.test_step_outputs], dim=0)
        precision = evaluate_with_conf_int(samples, f_precision, labels, conditions, num_bootstraps=1000, alpha=5)
        recall = evaluate_with_conf_int(samples, f_recall, labels, conditions, num_bootstraps=1000, alpha=5)
        f1 = evaluate_with_conf_int(samples, f_f1, labels, conditions, num_bootstraps=1000, alpha=5)        
        
        metrics = {
            'Threshold': threshold,
            'Precision': precision[0],
            'Precision_LB': precision[1][0],
            'Precision_UB': precision[1][1],
            'Recall': recall[0],
            'Recall_LB': recall[1][0],
            'Recall_UB': recall[1][1],
            'F1': f1[0],
            'F1_LB': f1[1][0],
            'F1_UB': f1[1][1]
        }

        # display results using pandas DataFrame
        m_names = list(metrics.keys())
        results = pd.DataFrame(np.array([[metrics[m_] for m_ in m_names]]), columns=m_names)
        print(results)

        # empty test step outputs list
        self.test_step_outputs = []

    def on_save_checkpoint(self, checkpoint):
        # Persist selected evaluation threshold so standalone test runs don't fall back to 0.5.
        checkpoint['eval_threshold'] = float(getattr(self, '_eval_threshold', self.hparams.decision_threshold_fixed))

    def on_load_checkpoint(self, checkpoint):
        # Restore persisted threshold when available.
        if checkpoint.get('eval_threshold', None) is not None:
            self._eval_threshold = float(checkpoint['eval_threshold'])
            # Keep fixed threshold in sync for explicit fixed-mode inference.
            self.hparams.decision_threshold_fixed = self._eval_threshold

        # because of code changes, early experiment checkpoints have the `model.resnet` attribute
        # the following code is to adapt the state dict to the current version of the code
        resnet_regex = re.compile('resnet.')
        feature_extractor_regex = re.compile('(model.embedder|model.encoder)')
        state_dict_keys = list(checkpoint['state_dict'].keys())
        if any([resnet_regex.search(state_dict_key) != None for state_dict_key in state_dict_keys]):
            for state_dict_key in state_dict_keys:
                new_state_dict_key = resnet_regex.sub('', state_dict_key)
                if (match := feature_extractor_regex.search(new_state_dict_key)):
                    new_state_dict_key = new_state_dict_key[:6] + 'feature_extractor.' + new_state_dict_key[6:]
                weights = checkpoint['state_dict'].pop(state_dict_key)
                checkpoint['state_dict'][new_state_dict_key] = weights                
