# Deep-Learning-APL745-
Basic Machine Learning Codes written from scratch in Python.

## [Non-Linear Regression](https://github.com/sob-ANN/Deep-Learning-APL745-/blob/main/Non-Linear%20Regression.ipynb)
Polynomial Regression applied to solve a simple problem of projectile motion. Use of Batch/Mini-batch/Stocastic Gradient Descent from scratch and plotting Number of Epochs vs Cost Function plots. Also, using a variable learning rate optimized by Line Search Algorithm

## [Multivariate Regression](https://github.com/sob-ANN/Deep-Learning-APL745-/blob/main/Multivariate%20Linear%20Regression.ipynb)
Multivariate Regression using Gradient Descent to predict house prices.

## [Binary Classification](https://github.com/sob-ANN/Deep-Learning-APL745-/blob/main/Binary%20Classification-Logistic%20Reg.ipynb)
Binary Classification of data containing multiple features which affect rainfall. Using that data we make predictions on a test set whether it will rain tomorrow or not.
We use logistic Regression as our hypothesis function and Binary Cross Entropy loss. Optimized using scipy library.

## [Multi-Class Classification](https://github.com/sob-ANN/Deep-Learning-APL745-/blob/main/OnevRest%20Classification.ipynb)
Multiclass Classification of MNIST Dataset from scratch. Optimization is done using Scipy Library. Binary Cross Entropy is the loss function used with Logistic Sigmoid as our hypothesis function. Target labels are one-hot encoded to be able to perform matrix operations.

## [Softmax Multi-Class Classification](https://github.com/sob-ANN/Deep-Learning-APL745-/blob/main/Softmax%20Classification.ipynb)
Multi-class Classification using softmax on the MNIST Fashion Dataset from scratch. Optimized using Batch Gradient Descent. Also, the accuracy is compared with the One-vs-Rest case.

## [Forward Problem using Physics-Informed-Neural-Network](https://github.com/sob-ANN/Deep-Learning-APL745-/blob/main/forward_problem_main.ipynb)
Physics Informed Neural Networks are a type of Universal Function Approximators that can be used to solve the underlying Differential Equation of a Physics problem. In this example, the deflection of a 1D Bar is modelled using PINN. Deflection at the two boundaries is zero and has been included in the modified loss function. In this case, no Input(x)-Output(displacement(x)) data is given. We have used only the Differential Equation of the underlying Physics and sampled points within the domain and minimised the 'residue' from the Differential Equation.

## [Inverse Problem using Physics-Informed-Neural-Network](https://github.com/sob-ANN/Deep-Learning-APL745-/blob/main/PINN_bar_inverse_main.ipynb)
In this example, we have solved an inverse problem of the same bar taken above. In this case, the deflections are known at each point on the bar. However, we do not know the physical properties of the bar(Axial Stiffness). Following similar principles as above, we have solved for EA(Axial Stiffness).
