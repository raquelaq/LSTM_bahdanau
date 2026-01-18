import argparse

def main():
    parser = argparse.ArgumentParser(description="Seq2Seq con atención de Bahdanau (traducción ES<->EN)")
    parser.add_argument("--mode", choices=["train", "infer"], required=True)
    parser.add_argument("--direction", choices=["es-en", "en-es"], default="es-en",
                        help="Solo aplica a infer. En train se entrenan ambas direcciones.")
    args = parser.parse_args()

    if args.mode == "train":
        from train import run_training
        run_training()
    else:
        from infer import run_inference
        run_inference(args.direction)

if __name__ == "__main__":
    main()
