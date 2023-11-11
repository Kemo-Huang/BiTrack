from pathlib import Path


def main():
    det_path = Path("data/tracking/virconv/")
    n = 0
    max_score = -100
    min_score = -100
    for seq in det_path.iterdir():
        for file_path in seq.iterdir():
            with open(file_path) as f:
                lines = f.readlines()
                n += len(lines)
                if len(lines) > 0:
                    max_score = max(
                        max_score, max([float(line.split()[-1]) for line in lines])
                    )
                    min_score = min(
                        min_score, min([float(line.split()[-1]) for line in lines])
                    )
    print(n, max_score, min_score)


if __name__ == "__main__":
    main()
