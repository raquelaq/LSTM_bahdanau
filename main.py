import argparse

def main():
    parser = argparse.ArgumentParser(description="Seq2Seq con atención de Bahdanau")
    parser.add_argument("--mode", choices=["train", "infer"], required=True)
    args = parser.parse_args()

    if args.mode == "train":
        import train  # ejecuta entrenamiento
    elif args.mode == "infer":
        from infer import run_inference
        run_inference()

if __name__ == "__main__":
    main()
