import torch

from av2unit.avhubert.hubert_pretraining import *
from av2unit.avhubert.hubert_dataset import *

class AVHubertUnitDataset(AVHubertDataset):
    def __init__(
            self,
            sample_rate: float,
            normalize: bool = False,
            stack_order_audio: int=1,
            image_mean: float=0,
            image_std: float=1,
            image_crop_size: int=88,
            image_aug: bool=False,
            modalities: Optional[List[str]]=None,
            noise_prob=0,
    ):
        self.audio_root = ""

        self.modalities = set(modalities)
        self.sample_rate = sample_rate
        self.stack_order_audio = stack_order_audio

        self.noise_prob = noise_prob

        self.normalize = normalize
        if image_aug:
            self.transform = custom_utils.Compose([
                custom_utils.Normalize( 0.0,255.0 ),
                custom_utils.RandomCrop((image_crop_size, image_crop_size)),
                custom_utils.HorizontalFlip(0.5),
                custom_utils.Normalize(image_mean, image_std) ])
        else:
            self.transform = custom_utils.Compose([
                custom_utils.Normalize( 0.0,255.0 ),
                custom_utils.CenterCrop((image_crop_size, image_crop_size)),
                custom_utils.Normalize(image_mean, image_std) ])
        logger.info(f"image transform: {self.transform}")

@register_task("av_hubert_unit_pretraining", dataclass=AVHubertPretrainingConfig)
class AVHubertUnitPretrainingTask(AVHubertPretrainingTask):
    def load_dataset(self) -> None:
        self.dataset = AVHubertUnitDataset(
            sample_rate=self.cfg.sample_rate,
            normalize=self.cfg.normalize,
            stack_order_audio=self.cfg.stack_order_audio,
            image_mean=self.cfg.image_mean,
            image_std=self.cfg.image_std,
            image_crop_size=self.cfg.image_crop_size,
            modalities=self.cfg.modalities,
        )
    def inference(self, model, sample):
        x, padding_mask = model.extract_finetune(**sample)

        label_embs_list = model.label_embs_concat.split(model.num_classes, 0)
        proj_x = model.final_proj(x)
        if model.untie_final_proj:
            proj_x_list = proj_x.chunk(len(model.num_classes), dim=-1)
        else:
            proj_x_list = [proj_x for _ in model.num_classes]
        logit_list = [model.compute_logits(proj, emb).view(-1, num_class) for proj, emb, num_class in zip(proj_x_list, label_embs_list, model.num_classes)] # [[B*T, V]]

        pred_even = logit_list[0].argmax(dim=-1).cpu()
        pred_odd = logit_list[1].argmax(dim=-1).cpu()
        pred = torch.stack([pred_even, pred_odd]).transpose(0,1).reshape(-1)

        return pred
