{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPpceOB6D1QZKns/ZyvJYJL",
      "include_colab_link": true
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
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/DaniChinwendu/EWC_ANN-MODEL/blob/main/Ewc_1_Streamlit.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "XTkjU66mZdpf",
        "outputId": "3bc12acd-284f-49ef-b91b-d093583f050a"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Writing app.py\n"
          ]
        }
      ],
      "source": [
        "\n",
        "%%writefile app.py\n",
        "import numpy as np\n",
        "from sklearn.preprocessing import MinMaxScaler\n",
        "import pickle\n",
        "import pandas as pd\n",
        "import streamlit as st \n",
        "#collect the model object file\n",
        "filename ='EWC_1.pkl' \n",
        "model = pickle.load(open(filename,'rb'))\n",
        "\n",
        "def welcome():\n",
        "    return \"Welcome All\"\n",
        "def inverse_transform(y_pred, min_max_values):\n",
        "  min_value = min_max_values[0]\n",
        "  max_value = min_max_values[1]\n",
        "  return (y_pred * (max_value - min_value)) + min_value\n",
        "\n",
        "def prediction_LOGEC3(log_DPRA_mean, log_hCLAT_MIT, scaler, model):\n",
        "    # Scale the input\n",
        "    scaled_input = scaler.transform([[log_DPRA_mean, log_hCLAT_MIT]])\n",
        "    prediction = model.predict(scaled_input)\n",
        "    return prediction\n",
        "\n",
        "def main():\n",
        "    st.title(\"'EDELWEISS CONNECT ITS SKIN SENSITIZATION SOLUTION'\")\n",
        "    st.markdown('An Artificial Neural Network Regression model Utilizing invitro and inchemo(h-CLAT,DPRA) Descriptors for predicting skin Sensitization')\n",
        "    html_temp = \"\"\"\n",
        "    EWC_1 SKIN SENSITIZATION PREDICTION App \n",
        "    \"\"\"\n",
        "    st.markdown(html_temp, unsafe_allow_html=True)\n",
        "    log_DPRA_mean = st.number_input(\"DPRA\",min_value=None, max_value=None, value=0.0, step=None,)\n",
        "    log_hCLAT_MIT = st.number_input(\"hCLAT\",min_value=None, max_value=None, value=0.0, step=None,)\n",
        "\n",
        "    \n",
        "    prediction_type = st.selectbox(\"Select prediction type:\", [\"Two-class\", \"Three-class\"])\n",
        "    if st.button(\"Predict\"):\n",
        "    # Scale the input\n",
        "      scaler = MinMaxScaler()\n",
        "      scaler.fit([[log_DPRA_mean, log_hCLAT_MIT]])\n",
        "    # Call the prediction function\n",
        "      result = prediction_LOGEC3(log_DPRA_mean, log_hCLAT_MIT, scaler, model)\n",
        "    # Convert the prediction back to the original scale\n",
        "      min_max_values=(0,1)\n",
        "      result = inverse_transform(result,min_max_values)#scaler.inverse_transform(result,min_max_values)\n",
        "      #result=result.reshape(1,1)\n",
        "      if result is not None:\n",
        "        if prediction_type == \"Three-class\":\n",
        "            if float(result) < (-1):\n",
        "                result = 'Strong'\n",
        "            elif float(result) >= (-1) and float(result) < 0:\n",
        "                result = 'Strong'\n",
        "            elif float(result) >= 0 and float(result) < 1:\n",
        "                result = 'Moderate'\n",
        "            elif float(result) >= 1:\n",
        "                result = 'Moderate'\n",
        "            else:\n",
        "                result = 'Non'\n",
        "        else:\n",
        "            if float(result) < (-1):\n",
        "                result = 'Positive'\n",
        "            elif float(result) >= (-1) and float(result) < 0:\n",
        "                result = 'Positive'\n",
        "            elif float(result) >= 0 and float(result) < 1:\n",
        "                result = 'Positive'\n",
        "            elif float(result) >= 1:\n",
        "                result = 'Positive'\n",
        "            else:\n",
        "                result = 'Negative'\n",
        "        st.success(f'The chemical Potency is {result}')\n",
        "    else:\n",
        "        st.warning(\"Prediction failed, please check your inputs and try again.\")\n",
        "    \n",
        "if __name__=='__main__':\n",
        "    main()\n",
        "      "
      ]
    }
  ]
}