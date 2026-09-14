# Proyecto Gimnasio — Servicio de IA y analítica

Servicio Python de generación online y procesos batch de análisis y machine learning.

## Responsabilidad

### Generación online

- exponer un OpenAPI versionado para backend;
- aceptar solicitudes asíncronas e idempotentes;
- orquestar el LLM alojado en el Polo;
- persistir estados y resultados en estructuras autorizadas de Neon;
- registrar modelo, configuración, contrato e instante;
- no crear ni aprobar rutinas.

### Analítica batch

- construir features point-in-time;
- entrenar y evaluar contra criterios simples;
- calcular resultados reproducibles;
- escribir salidas precalculadas para backend.

## Despliegue objetivo

API y worker se ejecutan en el Polo. Ngrok expone únicamente la API Python mediante un dominio estable. El LLM permanece local o privado y no accede a PostgreSQL.

## Requisitos actuales

- Python 3.13;
- acceso autorizado a Neon Test;
- acceso a la API del LLM del Polo para integración real.

## Inicio local

```bash
python -m venv .venv
# Activar el entorno virtual
python -m pip install -e ".[dev]"
cp .env.example .env

# API (recibe la solicitud, 202 inmediato)
python -m uvicorn gym_engine.api.app:app --reload

# Worker (procesa fuera de la petición HTTP, proceso separado)
python -m gym_engine.worker.poller
```

Backend es dueño de las migraciones; este repositorio no ejecuta cambios de esquema — las tablas `ai_generation_*` deben existir de antemano en `DATABASE_URL`.

## Verificación

```bash
python -m ruff check .
python -m mypy src
python -m pytest
```

La interfaz de datos con el backend está descrita en [docs/data-interface.md](docs/data-interface.md). El corpus funcional compartido está indexado en [docs/README.md](docs/README.md).
La interfaz con el backend está descrita en `architecture/data-interface.md` del [repositorio documental](https://github.com/Maico-Zurbriggen/proyecto-gimnasio-documentacion). Allí también viven el corpus funcional, la arquitectura y las reglas de dominio. Para trabajo asistido por IA, comenzar por su `AGENTS.md` y `manifest.json`.
