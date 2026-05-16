 
FROM python:3.14-slim AS builder 
  
# Install system build tools needed to compile some Python packages 
# --no-install-recommends: don't install optional packages (smaller image) 
# rm -rf /var/lib/apt/lists/*: delete apt cache after installing 
RUN apt-get update && apt-get install -y --no-install-recommends \ 
    build-essential \ 
    curl \ 
    git \ 
    && rm -rf /var/lib/apt/lists/* 
  
# Install Poetry (the same way we did on our laptops) 
ENV POETRY_HOME=/opt/poetry \ 
    POETRY_VIRTUALENVS_IN_PROJECT=true \ 
    POETRY_NO_INTERACTION=1 \ 
    POETRY_VERSION=1.8.3 
  
RUN curl -sSL https://install.python-poetry.org | python3 - 
ENV PATH="/opt/poetry/bin:$PATH" 
  
# Set working directory inside the builder stage 
WORKDIR /app 
  
# Copy ONLY the dependency files first 
# Why: Docker caches each layer. If only your code changes (not dependencies), 
#      Docker will reuse the cached 'poetry install' layer — much faster! 
COPY pyproject.toml poetry.lock ./ 
  
# Install ONLY production dependencies (no pytest, black, etc.) 
RUN poetry install --only=main --no-root 
  
# ══════════════════════════════════════════════════════ 
# STAGE 2: RUNTIME 
# Purpose: Lean, production-ready image 
# This is what gets deployed 
# ══════════════════════════════════════════════════════ 
FROM python:3.11-slim AS runtime 
  
# Security: create a non-root user 
# Running as root inside containers is a security risk 
RUN useradd --create-home --shell /bin/bash appuser 
  
# Install curl for the HEALTHCHECK command only 
RUN apt-get update && apt-get install -y --no-install-recommends curl \ 
    && rm -rf /var/lib/apt/lists/*
    WORKDIR /app 
  
# Copy the virtual environment from the BUILDER stage 
# We get all the installed packages WITHOUT Poetry or build tools 
COPY --from=builder /app/.venv /app/.venv 
  
# Set the PATH to use our virtual environment's Python 
ENV PATH="/app/.venv/bin:$PATH" \ 
    PYTHONDONTWRITEBYTECODE=1 \ 
    PYTHONUNBUFFERED=1 
# PYTHONDONTWRITEBYTECODE: don't create .pyc files (cleaner image) 
# PYTHONUNBUFFERED: print statements show immediately (better logs) 
  
# Copy application code 
# --chown: set file ownership to our non-root user 
COPY --chown=appuser:appuser . . 
  
# Switch to non-root user 
USER appuser 
  
# Document the port (doesn't actually open it — docker run -p does that) 
EXPOSE 8000 
  
# Health check: Docker will call this every 30s 
# If it fails 3 times, the container is marked 'unhealthy' 
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \ 
    CMD curl -f http://localhost:8000/health || exit 1 
  
# The command to run when the container starts 
# Using exec form ["..."] instead of shell form (more reliable signal handling) 
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 