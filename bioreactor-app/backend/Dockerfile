FROM eclipse-temurin:21-jre-alpine

RUN apk add --no-cache python3 py3-pip py3-numpy py3-scipy py3-psycopg2

WORKDIR /app

# Copy the jar
COPY target/bioreactor-app-0.1.0.jar app.jar

# Copy Python simulation engine
COPY python_sim/ /app/python_sim/

# Install Python deps
RUN pip3 install --no-cache-dir --break-system-packages numpy scipy psycopg2-binary cobra

EXPOSE 8080

ENTRYPOINT ["java", "-jar", "app.jar"]
