#!/usr/bin/env python3

# predictions_validation, predictions_test, description, labels_validation, labels_test
npz = np.load(Path("~/Downloads/cifar10_predictions.npz").expanduser())
test_vals = npz["predictions_test"]
test_labs = npz["labels_test"]
test_idx = test_vals[(~np.isnan(test_vals[:,:,:,0]))].reshape(30,15,5000,10)
