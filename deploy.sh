# replace "contabo" with your server name
rsync -av --exclude='.git' --exclude='.mypy_cache/' --exclude='.github/' --exclude='summaries/' --exclude='summary/' ../tts/ contabo:/root/summary