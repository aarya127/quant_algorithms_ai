#!/bin/sh
# Render Disk support
#
# On Render, a persistent disk is mounted at /app/mnt (configured in render.yaml).
# We redirect the three runtime data directories to that disk via symlinks so
# that model artefacts, feature CSVs, and MLflow data survive pod restarts and
# redeploys.
#
# On every other environment (local Docker, Kubernetes) /app/mnt won't exist
# and the block below is a no-op — paths fall back to the image directories.
if [ -d /app/mnt ]; then
    echo "[entrypoint] Render disk detected at /app/mnt — setting up persistent paths..."

    # Create subdirs on disk if they don't exist yet
    mkdir -p /app/mnt/data_pipelines /app/mnt/model_registry /app/mnt/mlruns

    # On first deploy, seed the model_registry AND data_pipelines from the image
    # so predictions work immediately without needing a pipeline run first.
    if [ ! -f /app/mnt/.seeded ]; then
        SRC=/app/algorithms/machine_learning_algorithms/supervised/model_registry
        if [ -d "$SRC" ] && [ "$(ls -A $SRC 2>/dev/null)" ]; then
            echo "[entrypoint] Seeding model_registry from image (first deploy)..."
            cp -r "$SRC/." /app/mnt/model_registry/
        fi
        DSRC=/app/algorithms/machine_learning_algorithms/data_pipelines
        if [ -d "$DSRC" ] && [ "$(ls -A $DSRC 2>/dev/null)" ]; then
            echo "[entrypoint] Seeding data_pipelines from image (first deploy)..."
            cp -r "$DSRC/." /app/mnt/data_pipelines/
        fi
        touch /app/mnt/.seeded
    fi

    # Redirect runtime dirs to the persistent disk via symlinks
    rm -rf /app/algorithms/machine_learning_algorithms/data_pipelines
    ln -sfn /app/mnt/data_pipelines \
            /app/algorithms/machine_learning_algorithms/data_pipelines

    rm -rf /app/algorithms/machine_learning_algorithms/supervised/model_registry
    ln -sfn /app/mnt/model_registry \
            /app/algorithms/machine_learning_algorithms/supervised/model_registry

    rm -rf /app/mlruns
    ln -sfn /app/mnt/mlruns /app/mlruns

    echo "[entrypoint] Persistent paths ready."
fi

exec gunicorn app:app \
    --bind 0.0.0.0:${PORT:-8080} \
    --workers ${GUNICORN_WORKERS:-2} \
    --worker-class sync \
    --timeout 120 \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    --access-logfile - \
    --error-logfile -
