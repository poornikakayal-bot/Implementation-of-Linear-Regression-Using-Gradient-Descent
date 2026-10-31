# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. .Load and preprocess the dataset by importing the required libraries, reading the CSV file, applying one-hot encoding to the categorical State column, and standardizing the input features and target values.

2.Prepare the data for training by separating the independent features (X) and target variable (y), then adding a bias (intercept) column to the feature matrix.

3.Train the Linear Regression model using Batch Gradient Descent by initializing the parameters, repeatedly updating them to minimize the Mean Squared Error (MSE) cost function, and plotting the cost convergence graph.

4.Predict the profit for a new startup by preprocessing the new input in the same way as the training data, applying the learned parameters to make a prediction, and converting the predicted value back to the original scale.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Poornika Umapathy
RegisterNumber: 212225060200 
*/
```
{ 
  "nbformat": 4, 
  "nbformat_minor": 0, 
  "metadata": { 
    "colab": { 
      "provenance": [] 
    }, 
    "kernelspec": { 
      "name": "python3", 
      "display_name": "Python 3" 
    }, 
    "language_info": { 
      "name": "python" 
    } 
  }, 
  "cells": [ 
    { 
      "cell_type": "code", 
      "execution_count": 1, 
      "metadata": { 
        "id": "eJje4znO4LKk" 
      }, 
      "outputs": [], 
      "source": [ 
        "import numpy as np\n", 
        "import pandas as pd\n", 
        "from sklearn.preprocessing import StandardScaler\n", 
        "def linear_regression(X1,y,learning_rate = 0.1, num_iters = 
1000):\n", 
        "    X = np.c_[np.ones(len(X1)),X1]\n", 
        "    theta = np.zeros(X.shape[1]).reshape(-1,1)\n", 
        "" 
      ] 
    }, 
    { 
      "cell_type": "code", 
      "source": [ 
        "def linear_regression(X1, y, learning_rate=0.1, 
num_iters=1000):\n", 
        "    X = np.c_[np.ones(len(X1)), X1]\n", 
        "    theta = np.zeros((X.shape[1], 1))\n", 
        "\n", 
        "    for _ in range(num_iters):\n", 
        "        predictions = X.dot(theta)\n", 
        "        errors = predictions - y\n", 
        "        theta -= learning_rate * (1/len(X1)) * 
X.T.dot(errors)\n", 
        "\n", 
        "    return theta" 
      ], 
      "metadata": { 
        "id": "Tf352pv95Y6p" 
      }, 
      "execution_count": 10, 
      "outputs": [] 
    }, 
    { 
      "cell_type": "code", 
      "source": [ 
        "data = pd.read_csv(\"exp_3_50_Startups.csv\")\n", 
        "\n", 
        "print(data.head())\n" 
      ], 
      "metadata": { 
        "colab": { 
          "base_uri": "https://localhost:8080/" 
        }, 
        "id": "Rg66UxhE5nLT", 
        "outputId": "a4733ba1-7026-4e95-93b9-7527b985e9a9" 
      }, 
      "execution_count": 13, 
      "outputs": [ 
        { 
          "output_type": "stream", 
          "name": "stdout", 
          "text": [ 
            "   R&D Spend  Administration  Marketing Spend       State     
Profit\n", 
            "0  165349.20       136897.80        471784.10    New York  
192261.83\n", 
            "1  162597.70       151377.59        443898.53  California  
191792.06\n", 
            "2  153441.51       101145.55        407934.54     Florida  
191050.39\n", 
            "3  144372.41       118671.85        383199.62    New York  
182901.99\n", 
            "4  142107.34        91391.77        366168.42     Florida  
166187.94\n" 
          ] 
        } 
      ] 
    }, 
    { 
      "cell_type": "code", 
      "source": [ 
        "X = data.iloc[:, :-2].values\n", 
        "X1 = X.astype(float)\n", 
        "\n", 
        "y = data.iloc[:, -1].values.reshape(-1, 1)\n", 
        "\n", 
        "print(X)" 
      ], 
      "metadata": { 
        "colab": { 
          "base_uri": "https://localhost:8080/" 
        }, 
        "id": "hHn9OmWp6ic_", 
        "outputId": "22091744-57d0-40c2-ff25-a3923f0f43a9" 
      }, 
      "execution_count": 14, 
      "outputs": [ 
        { 
          "output_type": "stream", 
          "name": "stdout", 
          "text": [ 
            "[[165349.2  136897.8  471784.1 ]\n", 
            " [162597.7  151377.59 443898.53]\n", 
            " [153441.51 101145.55 407934.54]\n", 
            " [144372.41 118671.85 383199.62]\n", 
            " [142107.34  91391.77 366168.42]\n", 
            " [131876.9   99814.71 362861.36]\n", 
            " [134615.46 147198.87 127716.82]\n", 
            " [130298.13 145530.06 323876.68]\n", 
            " [120542.52 148718.95 311613.29]\n", 
            " [123334.88 108679.17 304981.62]\n", 
            " [101913.08 110594.11 229160.95]\n", 
            " [100671.96  91790.61 249744.55]\n", 
            " [ 93863.75 127320.38 249839.44]\n", 
            " [ 91992.39 135495.07 252664.93]\n", 
            " [119943.24 156547.42 256512.92]\n", 
            " [114523.61 122616.84 261776.23]\n", 
            " [ 78013.11 121597.55 264346.06]\n", 
            " [ 94657.16 145077.58 282574.31]\n", 
            " [ 91749.16 114175.79 294919.57]\n", 
            " [ 86419.7  153514.11      0.  ]\n", 
            " [ 76253.86 113867.3  298664.47]\n", 
            " [ 78389.47 153773.43 299737.29]\n", 
            " [ 73994.56 122782.75 303319.26]\n", 
            " [ 67532.53 105751.03 304768.73]\n", 
            " [ 77044.01  99281.34 140574.81]\n", 
            " [ 64664.71 139553.16 137962.62]\n", 
            " [ 75328.87 144135.98 134050.07]\n", 
            " [ 72107.6  127864.55 353183.81]\n", 
            " [ 66051.52 182645.56 118148.2 ]\n", 
            " [ 65605.48 153032.06 107138.38]\n", 
            " [ 61994.48 115641.28  91131.24]\n", 
            " [ 61136.38 152701.92  88218.23]\n", 
            " [ 63408.86 129219.61  46085.25]\n", 
            " [ 55493.95 103057.49 214634.81]\n", 
            " [ 46426.07 157693.92 210797.67]\n", 
            " [ 46014.02  85047.44 205517.64]\n", 
            " [ 28663.76 127056.21 201126.82]\n", 
            " [ 44069.95  51283.14 197029.42]\n", 
            " [ 20229.59  65947.93 185265.1 ]\n", 
            " [ 38558.51  82982.09 174999.3 ]\n", 
            " [ 28754.33 118546.05 172795.67]\n", 
            " [ 27892.92  84710.77 164470.71]\n", 
            " [ 23640.93  96189.63 148001.11]\n", 
            " [ 15505.73 127382.3   35534.17]\n", 
            " [ 22177.74 154806.14  28334.72]\n", 
            " [  1000.23 124153.04   1903.93]\n", 
            " [  1315.46 115816.21 297114.46]\n", 
            " [     0.   135426.92      0.  ]\n", 
            " [   542.05  51743.15      0.  ]\n", 
            " [     0.   116983.8   45173.06]]\n" 
          ] 
        } 
      ] 
    }, 
    { 
      "cell_type": "code", 
      "source": [ 
        "scaler_X = StandardScaler()\n", 
        "scaler_y = StandardScaler()\n", 
        "\n", 
        "X1_Scaled = scaler_X.fit_transform(X1)\n", 
        "Y1_Scaled = scaler_y.fit_transform(y)\n", 
        "\n", 
        "print(X1_Scaled)" 
      ], 
      "metadata": { 
        "colab": { 
          "base_uri": "https://localhost:8080/" 
        }, 
        "id": "RSQVeAfj6ncD", 
        "outputId": "117f360a-478d-4b5f-b256-984aa60b1372" 
      }, 
      "execution_count": 15, 
      "outputs": [ 
        { 
          "output_type": "stream", 
          "name": "stdout", 
          "text": [ 
            "[[ 2.01641149e+00  5.60752915e-01  2.15394309e+00]\n", 
            " [ 1.95586034e+00  1.08280658e+00  1.92360040e+00]\n", 
            " [ 1.75436374e+00 -7.28257028e-01  1.62652767e+00]\n", 
            " [ 1.55478369e+00 -9.63646307e-02  1.42221024e+00]\n", 
            " [ 1.50493720e+00 -1.07991935e+00  1.28152771e+00]\n", 
            " [ 1.27980001e+00 -7.76239071e-01  1.25421046e+00]\n", 
            " [ 1.34006641e+00  9.32147208e-01 -6.88149930e-01]\n", 
            " [ 1.24505666e+00  8.71980011e-01  9.32185978e-01]\n", 
            " [ 1.03036886e+00  9.86952101e-01  8.30886909e-01]\n", 
            " [ 1.09181921e+00 -4.56640246e-01  7.76107440e-01]\n", 
            " [ 6.20398248e-01 -3.87599089e-01  1.49807267e-01]\n", 
            " [ 5.93085418e-01 -1.06553960e+00  3.19833623e-01]\n", 
            " [ 4.43259872e-01  2.15449064e-01  3.20617441e-01]\n", 
            " [ 4.02077603e-01  5.10178953e-01  3.43956788e-01]\n", 
            " [ 1.01718075e+00  1.26919939e+00  3.75742273e-01]\n", 
            " [ 8.97913123e-01  4.58678535e-02  4.19218702e-01]\n", 
            " [ 9.44411957e-02  9.11841968e-03  4.40446224e-01]\n", 
            " [ 4.60720127e-01  8.55666318e-01  5.91016724e-01]\n", 
            " [ 3.96724938e-01 -2.58465367e-01  6.92992062e-01]\n", 
            " [ 2.79441650e-01  1.15983657e+00 -1.74312698e+00]\n", 
            " [ 5.57260867e-02 -2.69587651e-01  7.23925995e-01]\n", 
            " [ 1.02723599e-01  1.16918609e+00  7.32787791e-01]\n", 
            " [ 6.00657792e-03  5.18495648e-02  7.62375876e-01]\n", 
            " [-1.36200724e-01 -5.62211268e-01  7.74348908e-01]\n", 
            " [ 7.31146008e-02 -7.95469167e-01 -5.81939297e-01]\n", 
            " [-1.99311688e-01  6.56489139e-01 -6.03516725e-01]\n", 
            " [ 3.53702028e-02  8.21717916e-01 -6.35835495e-01]\n", 
            " [-3.55189938e-02  2.35068543e-01  1.17427116e+00]\n", 
            " [-1.68792717e-01  2.21014050e+00 -7.67189437e-01]\n", 
            " [-1.78608540e-01  1.14245677e+00 -8.58133663e-01]\n", 
            " [-2.58074369e-01 -2.05628659e-01 -9.90357166e-01]\n", 
            " [-2.76958231e-01  1.13055391e+00 -1.01441945e+00]\n", 
            " [-2.26948675e-01  2.83923813e-01 -1.36244978e+00]\n", 
            " [-4.01128925e-01 -6.59324033e-01  2.98172434e-02]\n", 
            " [-6.00682122e-01  1.31053525e+00 -1.87861793e-03]\n", 
            " [-6.09749941e-01 -1.30865753e+00 -4.54931587e-02]\n", 
            " [-9.91570153e-01  2.05924691e-01 -8.17625734e-02]\n", 
            " [-6.52532310e-01 -2.52599402e+00 -1.15608256e-01]\n", 
            " [-1.17717755e+00 -1.99727037e+00 -2.12784866e-01]\n", 
            " [-7.73820359e-01 -1.38312156e+00 -2.97583276e-01]\n", 
            " [-9.89577015e-01 -1.00900218e-01 -3.15785883e-01]\n", 
            " [-1.00853372e+00 -1.32079581e+00 -3.84552407e-01]\n", 
            " [-1.10210556e+00 -9.06937535e-01 -5.20595959e-01]\n", 
            " [-1.28113364e+00  2.17681524e-01 -1.44960468e+00]\n", 
            " [-1.13430539e+00  1.20641936e+00 -1.50907418e+00]\n", 
            " [-1.60035036e+00  1.01253936e-01 -1.72739998e+00]\n", 
            " [-1.59341322e+00 -1.99321741e-01  7.11122474e-01]\n", 
            " [-1.62236202e+00  5.07721876e-01 -1.74312698e+00]\n", 
            " [-1.61043334e+00 -2.50940884e+00 -1.74312698e+00]\n", 
            " [-1.62236202e+00 -1.57225506e-01 -1.36998473e+00]]\n" 
          ] 
        } 
      ] 
    }, 
    { 
      "cell_type": "code", 
      "source": [ 
        "theta = linear_regression(X1_Scaled, Y1_Scaled)\n", 
        "\n", 
        "print(\"Theta values:\")\n", 
        "print(theta)" 
      ], 
      "metadata": { 
        "colab": { 
          "base_uri": "https://localhost:8080/" 
        }, 
        "id": "kcj6hagr6vKJ", 
        "outputId": "0e546089-9a0a-4dd5-c9b2-8e79d9d9840d" 
      }, 
      "execution_count": 16, 
      "outputs": [ 
        { 
          "output_type": "stream", 
          "name": "stdout", 
          "text": [ 
            "Theta values:\n", 
            "[[-4.43645121e-16]\n", 
            " [ 9.17579849e-01]\n", 
            " [-1.86404295e-02]\n", 
            " [ 8.26108671e-02]]\n" 
          ] 
        } 
      ] 
    }, 
    { 
      "cell_type": "code", 
      "source": [ 
        "new_data = np.array([[165349.2, 136897.8, 471784.1]])\n", 
        "\n", 
        "new_Scaled = scaler_X.transform(new_data)\n", 
        "\n", 
        "prediction = np.dot(np.append(1, new_Scaled[0]), theta)\n", 
        "\n", 
        "prediction = prediction.reshape(-1, 1)\n", 
        "\n", 
        "pre = scaler_y.inverse_transform(prediction)\n", 
        "\n", 
        "print(\"Predicted value:\", pre)" 
      ], 
      "metadata": { 
        "colab": { 
          "base_uri": "https://localhost:8080/" 
        }, 
        "id": "HT3jeo9861wE", 
        "outputId": "c50cf0a1-6ba6-4c84-e511-6e6b85b469a9" 
      }, 
      "execution_count": 17, 
      "outputs": [ 
        { 
          "output_type": "stream",
            "name": "stdout", 
          "text": [ 
            "Predicted value: [[192521.25289052]]\n" 
          ] 
        } 
      ] 
    } 
  ] 
}


## Output:
<img width="716" height="1600" alt="WhatsApp Image 2026-10-31 at 8 13 02 AM" src="https://github.com/user-attachments/assets/f20faaba-9f9c-4d20-a0d4-1c4bdeefadb6" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
