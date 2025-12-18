## Project Structure

The project consists of three main folders:

* **`backend/`** – FastAPI backend service
* **`client/`** – Dashboard frontend application
* **`traffic-controller/`** – OpenCV + YOLO-based traffic detection system

---

## Prerequisites

* Ensure **Docker** is installed on your system

  * If not, download and install Docker before proceeding.

---

## Running Backend & Frontend (Docker)

The backend and frontend are containerized using Docker.

From the **project root directory**, run:

```bash
docker compose up --build
```

This will start:

* Backend (FastAPI)
* Frontend (Dashboard)
* Supporting services (DB, Redis, etc.)

---

## Running Traffic Controller (YOLO + OpenCV)

The traffic controller must be run **locally (outside Docker)** because it requires direct camera access.

### Step 1: Navigate to the traffic controller directory

```bash
cd traffic-controller
```

---

### Step 2: Create a virtual environment

```bash
python3 -m venv venv
```

---

### Step 3: Activate the virtual environment

**macOS / Linux**

```bash
source venv/bin/activate
```

**Windows (Command Prompt / PowerShell)**

```bash
venv\Scripts\activate
```

---

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

---

### Step 5: Run the traffic controller

```bash
python3 traffic_controller.py
```

---

## Notes

* The traffic controller uses **OpenCV and YOLO**, which require direct access to the camera.
* Due to OS limitations, it is **not recommended to run the traffic controller inside Docker on macOS or Windows**.
* For deployment on **Linux-based systems** (Jetson, Raspberry Pi), the traffic controller can be containerized.