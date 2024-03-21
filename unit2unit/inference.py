import argparse
import numpy as np
import torch

from fairseq import checkpoint_utils, utils
from fairseq_cli.generate import get_symbols_to_strip_from_output

from unit2unit.task import UTUTPretrainingTask
from util import process_units, save_unit

def load_model(model_path, src_lang, tgt_lang, use_cuda=False):
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([model_path])

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    for model in models:
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()            
        model.prepare_for_inference_(cfg)

    task.source_language = src_lang
    task.target_language = tgt_lang

    generator = task.build_generator(
        models, cfg.generation
    )

    return task, generator

def main(args):
    use_cuda = torch.cuda.is_available() and not args.cpu

    task, generator = load_model(args.utut_path, args.src_lang, args.tgt_lang, use_cuda=use_cuda)

    with open(args.in_unit_path) as f:
        unit = list(map(int, f.readline().strip().split()))
    unit = task.source_dictionary.encode_line(
        " ".join(map(lambda x: str(x), process_units(unit, reduce=True))),
        add_if_not_exist=False,
        append_eos=True,
    ).long()
    unit = torch.cat([
        unit.new([task.source_dictionary.bos()]),
        unit,
        unit.new([task.source_dictionary.index("[{}]".format(task.source_language))])
    ])

    sample = {"net_input": {
        "src_tokens": torch.LongTensor(unit).view(1,-1),
    }}
    sample = utils.move_to_cuda(sample) if use_cuda else sample

    pred = task.inference_step(
        generator,
        None,
        sample,
    )[0][0]

    pred_str = task.target_dictionary.string(
        pred["tokens"].int().cpu(),
        extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator)
    )

    save_unit(pred_str, args.out_unit_path)

def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-unit-path", type=str, required=True, help="File path of source unit input"
    )
    parser.add_argument(
        "--out-unit-path", type=str, required=True, help="File path of target unit output"
    )
    parser.add_argument(
        "--utut-path", type=str, required=True, help="path to the UTUT pre-trained model"
    )
    parser.add_argument(
        "--src-lang", type=str, required=True,
        choices=["en","es","fr","it","pt"],
        help="source language"
    )
    parser.add_argument(
        "--tgt-lang", type=str, required=True,
        choices=["en","es","fr","it","pt"],
        help="target language"
    )
    parser.add_argument("--cpu", action="store_true", help="run on CPU")

    args = parser.parse_args()

    main(args)

if __name__ == "__main__":
    cli_main()
