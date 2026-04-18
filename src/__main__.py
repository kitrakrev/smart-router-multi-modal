"""Allow running the server with `python -m src`."""

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Smart Multi-Modal LLM Router")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    uvicorn.run("src.server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
