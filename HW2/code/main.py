import os
import matplotlib.pyplot as plt
from LogisticRegression import logistic_regression
from LRM import logistic_regression_multiclass
from DataReader import *
import sys

data_dir = "../data/"
train_filename = "training.npz"
test_filename = "test.npz"
    
def visualize_features(X, y):
    '''This function is used to plot a 2-D scatter plot of training features. 

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.

    Returns:
        No return. Save the plot to 'train_features.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
    plt.figure()
    plt.scatter(X[y==1][:,0], X[y==1][:,1], c='r', label='Class 1')
    plt.scatter(X[y==-1][:,0], X[y==-1][:,1], c='b', label='Class 2')
    plt.xlabel('Feature 1[Symmetry]')
    plt.ylabel('Feature 2[Intensity]')
    plt.title('2-D scatter plot of Training Features')
    plt.legend(loc='upper right', fontsize=12)
    plt.savefig('train_features.png', dpi=300)
    plt.show()
    ### END YOUR CODE



def visualize_result(X, y, W):
    '''Visualize logistic regression result with decision boundary.

    Args:
        X: Input feature matrix with shape [n_samples, 2]
        y: True labels with shape [n_samples,]. Class labels are either 1 or -1.
        W: Model weights (including bias) with shape [3,].

    Returns:
        A plot with the decision boundary.
    '''
    plt.figure()

    plt.scatter(X[y==1][:, 0], X[y==1][:, 1], c='r', label='Class 1')
    plt.scatter(X[y==-1][:, 0], X[y==-1][:, 1], c='b', label='Class 2')

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # Compute decision boundary: w^T x = 0 -> sigmoid(w1 * x1 + w2 * x2 + b) = 0.5
    grid_points = np.c_[np.ones((xx.ravel().shape[0], 1)), xx.ravel(), yy.ravel()]  # Add bias term
    Z = np.dot(grid_points, W)  
    Z = 1 / (1 + np.exp(-np.clip(Z, -100, 100)))  
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, levels=[0, 0.5, 1], colors=['blue', 'red'])
    
    plt.contour(xx, yy, Z, levels=[0.5], colors='k', linewidths=1, linestyles='dashed')  

    padding = 0.2  
    plt.xlim(X[:, 0].min() - padding, X[:, 0].max() + padding)
    plt.ylim(X[:, 1].min() - padding, X[:, 1].max() + padding)

    plt.xlabel('Feature 1[Symmetry]')
    plt.ylabel('Feature 2[Intensity]')
    plt.title('Logistic Regression Sigmoid Decision Boundary')
    plt.legend(loc='upper right', fontsize=12)

    # Save the plot as 'train_result_sigmoid.png'
    plt.savefig('train_result_sigmoid.png', dpi=300)
    plt.show()



def visualize_result_multi(X, y, W):
    """This function is used to plot the softmax model after training.

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 0,1,2.
        W: An array of shape [n_features, 3].

    Returns:
		No return. Save the plot to 'train_result_softmax.*' and include it
		in submission.
    """
    ### YOUR CODE HERE
    plt.figure()

    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='g', label='Class 0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='r', label='Class 1')
    plt.scatter(X[y == 2][:, 0], X[y == 2][:, 1], c='b', label='Class 2')

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))

    Z = np.dot(np.c_[np.ones((xx.ravel().shape[0], 1)), xx.ravel(), yy.ravel()], W)
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, levels=[-0.5, 0.5, 1.5, 2.5], colors=['green', 'red', 'blue'])
    # plt.contour(xx, yy, Z, colors='k', linewidths=1, linestyles='dashed')
    plt.contour(xx, yy, Z, levels=[0.5, 1.5], colors='k', linewidths=1, linestyles='dashed') 
    

    padding = 0.2  
    plt.xlim(X[:, 0].min() - padding, X[:, 0].max() + padding)
    plt.ylim(X[:, 1].min() - padding, X[:, 1].max() + padding)

    plt.xlabel('Feature 1[Symmetry]', fontsize=14)
    plt.ylabel('Feature 2[Intensity]', fontsize=14)
    plt.title('Logistic Regression Multi-Class Softmax Decision Boundary', fontsize=16)
    plt.legend(loc='upper right', fontsize=12)
    plt.savefig('train_result_softmax.png', dpi=300, bbox_inches='tight')
    plt.show()


def test_best_model(best_model):
    # Read and preprocess test data
    raw_test_data, test_labels = load_data(os.path.join(data_dir, test_filename))
    test_X_all = prepare_X(raw_test_data)
    test_y_all, test_idx = prepare_y(test_labels)

    test_X = test_X_all[test_idx]
    test_y = test_y_all[test_idx]

    # Set labels to 1 and -1
    test_y[test_y == 2] = -1

    # Evaluate test accuracy using the best model
    test_accuracy = best_model.score(test_X, test_y)
    print(f"Test Accuracy: {test_accuracy}")


def main():
	# ------------Data Preprocessing------------
	# Read data for training.
    sys.stdout = open('output.txt', 'w')
    
    raw_data, labels = load_data(os.path.join(data_dir, train_filename))
    raw_train, raw_valid, label_train, label_valid = train_valid_split(raw_data, labels, 2300)

    ##### Preprocess raw data to extract features
    train_X_all = prepare_X(raw_train)
    valid_X_all = prepare_X(raw_valid)
    ##### Preprocess labels for all data to 0,1,2 and return the idx for data from '1' and '2' class.
    train_y_all, train_idx = prepare_y(label_train)
    valid_y_all, val_idx = prepare_y(label_valid)  

    ####### For binary case, only use data from '1' and '2'  
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    ####### Only use the first 1350 data examples for binary training. 
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    ####### set lables to  1 and -1. Here convert label '2' to '-1' which means we treat data '1' as postitive class. 
	### YOUR CODE HERE
	#since class 1 remains 1 so we only modify class 2
    train_y[train_y == 2] = -1
    valid_y[valid_y == 2] = -1
	### END YOUR CODE
    data_shape = train_y.shape[0] 

    print("----------------------Q1 - Feature Visualization ----------------------")
    print()
#    # Visualize training data.
    visualize_features(train_X[:, 1:3], train_y)

    print("----------------------Q2/3 - Logistic Regression Sigmoid Case  ----------------------")
    print()
   # ------------Logistic Regression Sigmoid Case------------

   ##### Check BGD, SGD, miniBGD
    logisticR_classifier = logistic_regression(learning_rate=0.5, max_iter=100)

    logisticR_classifier.fit_BGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, data_shape)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_SGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, 1)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, 10)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    # Explore different hyper-parameters.
    ### YOUR CODE HERE
    def explore_hyperparameters():
        learning_rates = [0.01, 0.1, 0.5, 1]
        max_iters = [50, 100, 200]

        best_accuracy = 0
        best_model = None

        for lr in learning_rates:
            for iters in max_iters:
                # Initialize the model with specific hyperparameters
                logisticR_classifier = logistic_regression(learning_rate=lr, max_iter=iters)

                # Train with BGD
                logisticR_classifier.fit_BGD(train_X, train_y)
                accuracy = logisticR_classifier.score(train_X, train_y)

                # Update the best model if accuracy is better
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = logisticR_classifier

                print(f"Learning Rate: {lr}, Max Iterations: {iters}, Accuracy: {accuracy}")

        return best_model
    best_logisticR = explore_hyperparameters()
    print(f"Best Model - Accuracy: {best_logisticR.score(train_X, train_y)}")
    ### END YOUR CODE

	# Visualize the your 'best' model after training.
    visualize_result(train_X[:, 1:3], train_y, best_logisticR.get_params())


    # Use the 'best' model above to do testing. Note that the test data should be loaded and processed in the same way as the training data.
    ### YOUR CODE HERE
    test_best_model(best_logisticR)
    ### END YOUR CODE

    print("----------------------Q4 - Logistic Multiple-class Case  ----------------------")
    print()
    # ------------Logistic Regression Multiple-class case, let k= 3------------
    ###### Use all data from '0' '1' '2' for training
    train_X = train_X_all
    train_y = train_y_all
    valid_X = valid_X_all
    valid_y = valid_y_all

    #########  miniBGD for multiclass Logistic Regression
    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=100,  k= 3)
    logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, 10)
    print(logisticR_classifier_multiclass.get_params())
    print(logisticR_classifier_multiclass.score(train_X, train_y))

    # Explore different hyper-parameters.
    ### YOUR CODE HERE
    def explore_hyperparameters_multiclass():
        learning_rates = [0.01, 0.1, 0.5, 1]
        max_iters = [50, 100, 200]
        batch_sizes = [1, 10, 100]

        best_accuracy = 0
        best_model = None

        for lr in learning_rates:
            for iters in max_iters:
                for batch_size in batch_sizes:
                    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=lr, max_iter=iters, k=3)

                    # Train with miniBGD
                    logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, batch_size)
                    accuracy = logisticR_classifier_multiclass.score(train_X, train_y)

                    # Update the best model if accuracy is better
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model = logisticR_classifier_multiclass

                    print(f"Learning Rate: {lr}, Max Iterations: {iters}, Batch Size: {batch_size}, Accuracy: {accuracy}")

        return best_model
    
    best_logistic_multi_R = explore_hyperparameters_multiclass()
    print(f"Best Model - Accuracy: {best_logistic_multi_R.score(train_X, train_y)}")
    ### END YOUR CODE

	# Visualize the your 'best' model after training.
    visualize_result_multi(train_X[:, 1:3], train_y, best_logistic_multi_R.get_params())


    # Use the 'best' model above to do testing.
    ### YOUR CODE HERE
    test_best_model(best_logistic_multi_R)
    ### END YOUR CODE

    print("----------------------Q5 - Connection between sigmoid and softmax ----------------------")
    print()
    # ------------Connection between sigmoid and softmax------------
    ############ Now set k=2, only use data from '1' and '2' 

    #####  set labels to 0,1 for softmax classifer
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    train_y[np.where(train_y==2)] = 0
    valid_y[np.where(valid_y==2)] = 0  
    
    ###### First, fit softmax classifer until convergence, and evaluate 
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE
    # For Softmax classifier, set labels to 0 and 1 (instead of 1 and -1)
    train_y_softmax = np.where(train_y == 2, 0, train_y)  # '2' -> '0'
    valid_y_softmax = np.where(valid_y == 2, 0, valid_y)  # '2' -> '0'

    # Train Softmax classifier
    softmax_classifier = logistic_regression_multiclass(learning_rate=0.5, max_iter=10000, k=2)
    softmax_classifier.fit_miniBGD(train_X, train_y_softmax, 10)

    # Evaluate Softmax classifier on the validation set
    softmax_accuracy = softmax_classifier.score(valid_X, valid_y_softmax)
    # print(f"Softmax Classifier Validation Accuracy: {softmax_accuracy}")
    ### END YOUR CODE


    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    #####       set lables to -1 and 1 for sigmoid classifer
	### YOUR CODE HERE
    # For Sigmoid: We treat '1' as positive and '2' as negative.
    train_y_sigmoid = np.where(train_y == 2, -1, train_y)  # '2' -> '-1'
    valid_y_sigmoid = np.where(valid_y == 2, -1, valid_y)  # '2' -> '-1'
	### END YOUR CODE 

    ###### Next, fit sigmoid classifer until convergence, and evaluate
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE
    


    # Train Sigmoid classifier
    sigmoid_classifier = logistic_regression(learning_rate=0.5, max_iter=10000)
    sigmoid_classifier.fit_SGD(train_X, train_y_sigmoid)

    # Evaluate Sigmoid classifier on the validation set
    sigmoid_accuracy = sigmoid_classifier.score(valid_X, valid_y_sigmoid)
    # print(f"Sigmoid Classifier Validation Accuracy: {sigmoid_accuracy}")


    ### END YOUR CODE
    # Output insights
    print(f"Softmax Classifier Accuracy: {softmax_accuracy}")
    print(f"Sigmoid Classifier Accuracy: {sigmoid_accuracy}")
    

    # visualize_result_multi(train_X[:, 1:3], train_y_softmax, softmax_classifier.get_params())

    # visualize_result(train_X[:, 1:3], train_y_sigmoid, sigmoid_classifier.get_params())




    ################Compare and report the observations/prediction accuracy


    # ------------End------------
    

if __name__ == '__main__':
	main()
    
    
