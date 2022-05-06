from resources import Detector


def main():
    with Detector() as detector:
        detector.init_model()

        detector.detect()


if __name__ == '__main__':
    main()
