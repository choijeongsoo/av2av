import logging
from fairseq.tasks import register_task
from fairseq.tasks.multilingual_denoising import MultilingualDenoisingConfig, MultilingualDenoisingTask

logger = logging.getLogger(__name__)

@register_task("utut_pretraining", dataclass=MultilingualDenoisingConfig)
class UTUTPretrainingTask(MultilingualDenoisingTask):
    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
    ):
        lang_list = self.cfg.langs.split(",")

        lang_token_ids = {
            self.dictionary.index("[{}]".format(lang))
            for lang in lang_list
        }

        if extra_gen_cls_kwargs is None:
            extra_gen_cls_kwargs = {}

        extra_gen_cls_kwargs["symbols_to_strip_from_output"] = lang_token_ids

        extra_gen_cls_kwargs["eos"] = self.dictionary.index("[{}]".format(self.target_language))

        extra_gen_cls_kwargs["tokens_to_suppress"] = [
            "[{}]".format(lang) for lang in lang_list if lang != self.target_language
        ] + [self.dictionary[self.mask_idx]]

        return super().build_generator(
            models,
            args,
            seq_gen_cls=seq_gen_cls,
            extra_gen_cls_kwargs=extra_gen_cls_kwargs,
        )
