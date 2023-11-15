# Beavs.ai - AI Hub

## About

This is the AI Hub for the Beavs.ai project. This is where we'll store our data, integrations, and other AI-related things.

## Getting Started

### Prerequisites

- [Python 3.9](https://www.python.org/downloads/release/python-390/)
- [Pip](https://pip.pypa.io/en/stable/getting-started/)
- [Uvicorn](https://www.uvicorn.org/)
- [FastAPI](https://fastapi.tiangolo.com/)

### Installation

1. Clone the repo

```sh
git clone https://github.com/OSU-App-Club/beavsAI-AI.git
```

2. Create a virtual enviornment for the project
```sh
cd beavsAI-AI
cd server
python -m venv venv
```

3. Activate the virtual enviornment
```sh
source venv/bin/activate
```

4. Install the required packages
```sh
pip install -r requirements.txt
```

5. Start the development server (with hot reloading)
```sh
uvicorn main:app --reload
```

6. Open the development server in your browser
```sh
http://localhost:8000
```

## Usage

### Data

- The `/data` directory is where we'll store our data. This will be used for indexing and loading our data.

### Server

- The `/server` directory is where we'll store our server code. This will be used for serving our API and other server-related things.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

This project is being developed by the [Oregon State University App Development Club](https://osuapp.club)