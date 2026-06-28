# =============================================================================
# Custom Airflow image for the CS611 Assignment 2 ML pipeline.
# Extends the official Airflow image and adds:
#   - OpenJDK 17 (required by PySpark 3.5.x to launch its local JVM)
#   - The project's Python ML dependencies (pandas, scikit-learn, xgboost, ...)
# =============================================================================
FROM apache/airflow:2.9.3-python3.11

# ---- System-level dependencies (need root) ---------------------------------
USER root

# Install a JDK so PySpark can launch its local JVM, plus procps for Spark.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        openjdk-17-jdk-headless \
        procps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    # Create an architecture-independent symlink so JAVA_HOME is stable
    # whether the image is built on amd64 or arm64 (Apple Silicon).
    && ln -s "$(dirname "$(dirname "$(readlink -f "$(which java)")")")" /usr/lib/jvm/current-java

# Expose JAVA_HOME for PySpark.
ENV JAVA_HOME=/usr/lib/jvm/current-java
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# ---- Python dependencies (must run as the airflow user) --------------------
USER airflow

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
