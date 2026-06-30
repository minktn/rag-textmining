# rag-textmining

## RUN DOCKER
```
docker-compose up -d --build
docker exec -it rag-textmining-api /bin/bash
```

## DIRECTORIES
```
project/
├── workspace/
│	├── data/
│	│	├── chunked/
│	│	│	└── landlaw_chunks.json		# Run ingest.py to generate this file
│	│	└── original/
│	│		└── landlaw.md				# Given data (download from Drive)
│   ├── scripts/
│   └── src/
├── .env
└── README.md
```