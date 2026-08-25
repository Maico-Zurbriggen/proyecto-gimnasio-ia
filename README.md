# Proyecto Gimnasio — Motor analítico

Procesos batch de análisis y machine learning de la plataforma de entrenamiento asistido.

## Responsabilidad

- construir features point-in-time;
- entrenar y evaluar modelos contra criterios de referencia simples;
- calcular scores y diagnósticos reproducibles;
- escribir resultados precalculados consumidos por el backend.

Este repositorio no expone HTTP y nunca entra en el camino de una petición del usuario.

## Requisitos

- Python 3.13;
- acceso a un PostgreSQL local o a snapshots sintéticos acordados.

## Inicio local

```bash
python -m venv .venv
# Activar el entorno virtual según el sistema operativo
python -m pip install -e ".[dev]"
cp .env.example .env
```

## Verificación

```bash
python -m ruff check .
python -m mypy src
python -m pytest
```

La interfaz con el backend está descrita en `architecture/data-interface.md` del [repositorio documental](https://github.com/Maico-Zurbriggen/proyecto-gimnasio-documentacion). Allí también viven el corpus funcional, la arquitectura y las reglas de dominio. Para trabajo asistido por IA, comenzar por su `AGENTS.md` y `manifest.json`.
