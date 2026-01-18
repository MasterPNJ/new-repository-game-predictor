import uvicorn
from .routes import app

def main():
    uvicorn.run(app, host="0.0.0.0", port=5656)

if __name__ == "__main__":
    main()