# DoctelliGen
### AI/LLM Powered Document Query

A web application that allows you to upload documents (pdf, text and md currently) and then use a Large Language Model (LLM) to query the documents and receive relevant answers.

## Example

Document Upload Dialog:

![doc upload dialog](media/upload_screen.png)

Document Query Dialog:

![doc query dialog](media/query_screen.png)


## Installation 

Create an activate a new Python environment:

```bash
conda create -n DoctelliGen python=3.11
conda activate DoctelliGen
```

Clone the repo and enter it:

```bash
git clone https://github.com/robert-mcdermott/DoctelliGen.git
cd DoctelliGen
```

Install the dependencies (install the latest version of poetry if you don't have it):

```
poetry install
```

Run the app:

```
python app.py
```