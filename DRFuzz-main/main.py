import importlib

from src.DrFuzz import DrFuzz
from src.deephunter import DeepHunter
from src.experiment_builder import get_experiment
from src.utility import merge_object
import matplotlib.pyplot as plt

def load_params(params):
    for params_set in params.params_set:
        m = importlib.import_module("params." + params_set)
        print(m)
        new_params = getattr(m, params_set)
        params = merge_object(params, new_params)
    return params


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Experiments Script For DeepReFuzz")
    parser.add_argument("--params_set", nargs='*', type=str, default=["alexnet", "fm", "kmn", "deephunter"],
                        help="see params folder")
    parser.add_argument("--dataset", type=str, default="FM", choices=["MNIST", "CIFAR10", "FM", "SVHN"])
    parser.add_argument("--model", type=str, default="Alexnet_prune", choices=["vgg16", "resnet18", "LeNet5", "Alexnet",
                                                                               "vgg16_adv_bim", "vgg16_adv_cw",
                                                                               "vgg16_apricot",
                                                                               "LeNet5_adv_bim", "LeNet5_adv_cw",
                                                                               "LeNet5_apricot",
                                                                               "Alexnet_adv_bim", "Alexnet_adv_cw",
                                                                               "Alexnet_apricot",
                                                                               "resnet18_adv_bim", "resnet18_adv_cw",
                                                                               "resnet18_apricot",
                                                                               "LeNet5_quant", "vgg16_quant",
                                                                               "resnet18_quant", "Alexnet_quant",
                                                                               "LeNet5_prune", "vgg16_prune",
                                                                               "resnet18_prune", "Alexnet_prune"])

    parser.add_argument("--coverage", type=str, default="kmn", choices=["change", "neuron", "kmn", "nbc", "snac"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--time", type=int, default=1440)
    params = parser.parse_args()

    print(params)
    params = load_params(params)
    params.time_minutes = params.time
    time_h=int(params.time_minutes/60)
    print("TTTTTTTTTTTTTTTTTTTTTTTT", time_h)
    params.time_period = params.time_minutes * 60
    experiment = get_experiment(params)
    experiment.time_list = [i * 30 for i in range(1, params.time // 30 + 1 + 1)]

    if params.framework_name == 'drfuzz':
        dh = DrFuzz(params, experiment)
    elif params.framework_name == 'deephunter':
        dh = DeepHunter(params, experiment)
    else:
        raise Exception("No Framework Provided")

    import numpy as np
    import os

    print(os.path.abspath(__file__))
    experiment_dir = str(params.coverage)
    dir_name = 'experiment_' + str(params.framework_name)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    if not os.path.exists(os.path.join(dir_name, experiment_dir)):
        os.mkdir(os.path.join(dir_name, experiment_dir))
    Version_S = input("Enter the number of your desired scenario: DNN1/2>>1, DNN3/4>>2, DNN4/5>>3, DNN7/8>>4::::")
    num=input("Enter the number of run: 1 or 2, 3")
    print("You entered version:", Version_S, "and run number", num)
    both_fail, regression_faults, weaken, triggering_fault, bf_l, tf_l, rf_l= dh.run()

   
    # np.save(os.path.join(dir_name, experiment_dir, "bothfail.npy"), np.asarray(both_fail))
    # np.save(os.path.join(dir_name, experiment_dir, "regression_faults.npy"), np.asarray(regression_faults))
    # np.save(os.path.join(dir_name, experiment_dir, "weaken.npy"), np.asarray(weaken))
    print('TOTAL Triggering:', len(triggering_fault))
    print('TOTAL BOTH (M1!=T and M2!=T):', len(both_fail))
    print('TOTAL REGRESSION:', len(regression_faults), np.asarray(regression_faults).shape)
    print('TOTAL WEAKEN:', len(weaken))
    print('CORPUS', dh.corpus)
    # np.save(os.path.join(dir_name, experiment_dir, "corpus.npy"), np.asarray(dh.corpus_list))
    ffi = f"/content/drive/MyDrive/Paper_Diff_testing/DRFuzz-main/Failures_images/{params.dataset}_S{Version_S}/time_{time_h}h/run{num}"
    print(ffi+"/triggering_f.npy")
    # np.save(os.path.join(dir_name, experiment_dir, "/bothfail.npy"), np.asarray(both_fail))
    # np.save(os.path.join(dir_name, experiment_dir, "/regression_faults.npy"), np.asarray(regression_faults))
    np.save( ffi+"/triggering_f.npy", np.asarray(triggering_fault))
    # np.save( ffi+ "/bothfail.npy", np.asarray(both_fail))
    # np.save( ffi+ "/regression_faults.npy", np.asarray(regression_faults))
    print('ITERATION', dh.experiment.iteration)

    if params.framework_name == 'drfuzz':
        print('SCORE', dh.experiment.coverage.get_failure_type())
    elif params.framework_name == 'deephunter':
        print('SCORE', dh.experiment.coverage.get_current_coverage())

    import matplotlib.pyplot as plt

    plt.imshow(regression_faults[0].input)
    plt.show()
    
    def save_failure_image(failure_case, label_info, category, params, Version_S, i):
        m1,m2, gt = label_info
        image = failure_case.ref_image

        # Check and adjust the shape of the image
        if image.shape[-1] == 1:  # Grayscale image (28, 28, 1)
            image = np.squeeze(image, axis=-1)  # Squeeze to make it (28, 28)
            cmap = 'gray'  # Use grayscale colormap
        elif image.shape[-1] == 3:  # RGB image (32, 32, 3)
            cmap = None  # No colormap needed for RGB
        else:
            raise ValueError("Invalid image shape")
         # Convert image to uint8
        if image.dtype == np.int16:
            image = image.astype(np.uint8)
        # Normalize floating-point images to range 0 to 1
        elif np.issubdtype(image.dtype, np.floating):
            image = (image - image.min()) / (image.max() - image.min())
          # plt.imsave(filename, image, cmap=cmap, format='png')
        arr.append(image)


        filename = f"/content/drive/MyDrive/Paper_Diff_testing/DRFuzz-main/Failures_images/{params.dataset}_S{Version_S}/time_{time_h}h/run{num}/{category}_M1_{m1}_M2_{m2}_GT{gt}_im_{i}.png"
        # plt.imsave(filename, image, cmap=cmap, format='png')
        return arr
     
    # Process and save images from regression_faults
    # for i in range(len(regression_faults)):
    #     save_failure_image(regression_faults[i], rf_l[i], 'RF', params, Version_S, i)
    arr=[]
    # Process and save images from triggering_fault
    for i, case in enumerate(triggering_fault):
         
        comp_a=save_failure_image(case, tf_l[i], 'TF', params, Version_S, i)


# Convert the list to a NumPy array with shape (N, 32, 32, 3)
    triggering_images_arr = np.array(comp_a)
    print("SSSSSSSS",triggering_images_arr.shape )
    np.save(f"/content/drive/MyDrive/Paper_Diff_testing/DRFuzz-main/Failures_images/{params.dataset}_S{Version_S}/time_{time_h}h/run{num}/trigg.npy", triggering_images_arr)

    # # Process and save images from both_fail
    # for i, case in enumerate(both_fail):
    #     save_failure_image(case, bf_l[i], 'BF', params, Version_S, i)
    
# bf_initial_lab,rf_initial_lab,tf_initial_lab
