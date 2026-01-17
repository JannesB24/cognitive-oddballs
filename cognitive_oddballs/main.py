from models.change_point_nassar_2016 import ChangePointNassarModel

def main():
    pass

    nassar_model = ChangePointNassarModel(X=df["x"], sigma_sequence=df["sigma"])
    normative_model_results = nassar_model.run()
    print(normative_model_results)


if __name__ == "__main__":
    main()
