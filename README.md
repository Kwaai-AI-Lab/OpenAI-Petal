<h1 align="center">OpenAI api compliant server for Petals distributed inference 👋</h1>
<p>
  <img alt="Version" src="https://img.shields.io/badge/version-1.0.0-blue.svg?cacheSeconds=2592000" />
  <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank">
    <img alt="License: CC-BY-4.0" src="https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg" />
  <a href="https://kwaaiailab.slack.com" target="_blank">
    <img alt="Slack: Kwaai.org" src="https://img.shields.io/badge/slack-join-green?logo=slack" />
  </a>  
  <img alt="Python" src="https://img.shields.io/badge/python-3.10-blue" />
  <img alt="Browser" src="https://img.shields.io/badge/Browser-chrome-red" />
</p>


OpenAI compliant api server to support following api calls and bridge to Petals [https://github.com/bigscience-workshop/petals] v1/generate

v1/models
v1/completions
v1/chat/completions



The best way to support TruLens is to give us a ⭐ on [GitHub](https://github.com/KWAAI-ai-lab/paiassistant) and join our [slack community](https://kwaaiailab.slack.com)!


### Installation and Setup
The steps below can be used to setup the enviroment for this project.
Alternatively you can setup the python3.10 environment.
The install will run with or without GPU. If you are running a private swarm node, you might need some gpu support to share the load with community inference servers. This project needs some resources for the tokenizer part of inference. It will run on cpu or gpu supported machines.

Note: The default setup and run process provide here will allow you to connect to petals public swarm. Data you send will be public. Please be aware!!

### Prerequisites.
1. Install petals package by following instructions at [https://github.com/bigscience-workshop/petals]
typically this step should work.
```bash
    pip install git+https://github.com/bigscience-workshop/petals
```
### Clone repository
1. clone this repository using git clone.

### Run api server using public peers at health.petals.dev
```bash
    cd OpenAI-Petal
    export PETALS_IGNORE_DEPENDENCY_VERSION=1
    uvicorn app_openai_json:app --host 0.0.0.0 --port 8000
```

### Launching private swarm

Follow steps at [https://github.com/bigscience-workshop/petals/wiki/Launch-your-own-swarm]
Set INITIAL_PEERS  to point to your private swarm.

#### Demo and tests
1. http://localhost:8000/docs in chrome browser to see the FastAPI api documentation and try out.
2. Use provided Petal_inference_webpage.html to test web access using chrome.
3. Use provided Petal-inference-Langchain-openai.py using python to test langchain supported methods using OpenAI and ChatOpenAI

## 📝 License

This project is [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) licensed.

