from easyeditor import MENDHyperParams, SERACHparams
from easyeditor import EditTrainer
from easyeditor import ZsreDataset
import argparse

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument("--hparams_file", required=True, type=str)
    parser.add_argument("--train_data_file", required=True, type=str)
    parser.add_argument("--val_data_file", required=True, type=str)
    parser.add_argument("--save_dir", required=True, type=str)
    
    # parser.add_argument("--backbone", type=str, default="chinese_llama7b")
    # parser.add_argument("--source_lang", type=str, default="en")
    return parser.parse_args()

def main(args):
    if args.editing_method == "MEND":
        training_hparams = MENDHyperParams
    elif args.editing_method == "SERAC":
        training_hparams = SERACHparams
    else:
        raise NotImplementedError()

    hparams = training_hparams.from_hparams(args.hparams_file)

    if hasattr(hparams, "results_dir"):
        hparams.results_dir = args.save_dir
    # if hasattr(hparams, "model_parallel"):
    #     hparams.model_parallel = True

    # print(hparams.results_dir)

    train_ds = ZsreDataset(args.train_data_file, config=hparams)
    eval_ds = ZsreDataset(args.val_data_file, config=hparams)
    # #     raise NotImplementedError()

    trainer = EditTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    trainer.run()

if __name__ == '__main__':
    main(parser_args())