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
```

Las dependencias HTTP y los comandos de API/worker se incorporarán con el esqueleto de integración. Backend es dueño de las migraciones; este repositorio no ejecuta cambios de esquema.

## Verificación

```bash
python -m ruff check .
python -m mypy src
python -m pytest
```

<<<<<<< Updated upstream
La interfaz de datos con el backend está descrita en [docs/data-interface.md](docs/data-interface.md). El corpus funcional compartido está indexado en [docs/README.md](docs/README.md).
=======
La arquitectura y contratos de persistencia están en [proyecto-gimnasio-documentacion](https://github.com/Maico-Zurbriggen/proyecto-gimnasio-documentacion). Para trabajo asistido por IA, comenzar por su `AGENTS.md` y `manifest.json`.
>>>>>>> Stashed changes
