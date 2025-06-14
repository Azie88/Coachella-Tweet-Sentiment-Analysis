# Coachella-Tweet-Sentiment-Analysis 🎉

<a href="https://sproutsocial.com/insights/twitter-sentiment-analysis/">
  <img src="https://github.com/user-attachments/assets/a157edd0-6aa4-48d4-a157-dd7844647845" alt="Twitter Sentiment Analysis" width="800"/>
</a>

This project is about Natural Language Processing, specifically text classification (Sentiment analysis). In this project, we will fine tune a RoBERTa base sentiment analysis model on tweets about the Coachella 2015 music festival lineup to create a model that can predict of the sentiment expressed in a tweet (e.g: neutral, positive, negative), then create gradio app to use the models and deploy the app on HuggingFace spaces. This project will use [Google Colab](https://colab.research.google.com/) to leverage the GPU computational power.

Read more about [Text classification with Hugging Face](https://huggingface.co/tasks/text-classification)

## Project Links 📑

- [Sentiment_Analysis_Coachella.ipynb](https://github.com/Azie88/Coachella-Tweet-Sentiment-Analysis/blob/main/Sentiment_Analysis_Coachella.ipynb): Model Fine Tuning process.
- [main.py](https://github.com/Azie88/Coachella-Tweet-Sentiment-Analysis/blob/main/main.py): Gradio app
- [Data](https://github.com/Azie88/Coachella-Tweet-Sentiment-Analysis/tree/main/Dataset): Folder with training and testing datasets for model development.
- [Hugging Face Space](https://huggingface.co/spaces/Azie88/Coachella-Tweet-Sentiment-Analysis): Gradio app deployed on Huggingface spaces

## Getting Started 🏁

You need to have [`Python 3`](https://www.python.org/) on your system. Then you can clone this repo and being at the repo's `root :: repository_name> ...`

1. Clone this repository: `git clone https://github.com/Azie88/Coachella-Tweet-Sentiment-Analysis.git`
2. On your IDE, create A Virtual Environment and Install the required packages for the project:

- Windows:
        
        python -m venv venv; 
        venv\Scripts\activate; 
        python -m pip install -q --upgrade pip; 
        python -m pip install -qr requirements.txt  

- Linux & MacOs:
        
        python3 -m venv venv; 
        source venv/bin/activate; 
        python -m pip install -q --upgrade pip; 
        python -m pip install -qr requirements.txt  

The two long command-lines have the same structure. They pipe multiple commands using the symbol ` ; ` but you can manually execute them one after the other.

- **Create the Python's virtual environment** that isolates the required libraries of the project to avoid conflicts;
- **Activate the Python's virtual environment** so that the Python kernel & libraries will be those of the isolated environment;
- **Upgrade Pip, the installed libraries/packages manager** to have the up-to-date version that will work correctly;
- **Install the required libraries/packages** listed in the `requirements.txt` file so that they can be imported into the python script and notebook without any issue.

**NB:** For MacOs users, please install `Xcode` if you have an issue.

3. Run the Gradio app (being at the repository root):

  Gradio: 
  
    For development

      gradio main.py
    
    For normal deployment/execution

      python main.py  

  - Go to your browser at the following address :
        
      http://localhost:7860

4. Run the jupyter notebook on colab for more indepth insights on the deep learning process.

## App Screenshots 🖼️

<table>
    <tr>
        <th> Gradio App on Huggingface spaces</th>
    </tr>
    <tr>
        <td><img src="Screenshots\Positive_Sentiment.png"/></td>
    </tr>
    <tr>
        <td><img src="Screenshots\Neutral_Sentiment.png"/></td>
    </tr>
    <tr>
        <td><img src="Screenshots\Negative_Sentiment.png"/></td>
    </tr>
</table>

## Contributions :handshake:

Open an issue, submit a pull request or feel free to fork this repository to make any improvements you have in mind.

## Author✍️

Andrew Obando

<a href="https://www.linkedin.com/in/andrewobando/"><img align="left" src="https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white" alt="Andrew Obando | LinkedIn"/></a>
<a href="https://medium.com/@obandoandrew8">
![Medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)
</a>

---

Feel free to star ⭐ this repository if you find it helpful!
