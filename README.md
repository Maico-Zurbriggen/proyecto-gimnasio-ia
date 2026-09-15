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

API FastAPI y consumidor durable se despliegan en Vercel. Vercel Queues desacopla la aceptación `202` del trabajo de generación. El LLM permanece en el Polo detrás de un dominio HTTPS estable de Cloudflare Tunnel protegido con un token Bearer; Ollama no se expone sin autenticación.

## Requisitos actuales

- Python 3.13;
- acceso autorizado a Neon Test;
- acceso a Neon con el rol restringido de IA;
- acceso al endpoint autenticado del LLM del Polo para integración real.

## Inicio local

```bash
python -m venv .venv
# Activar el entorno virtual
python -m pip install -e ".[analytics,dev]"
cp .env.example .env
python -m uvicorn app:app --reload --port 8000
```

En PowerShell, usar `Copy-Item .env.example .env`. Backend es dueño de las migraciones; este repositorio no ejecuta cambios de esquema.

Endpoints:

- `GET /health`: salud del proceso, público y sin consultar dependencias;
- `GET /ready`: verifica Neon y el LLM; requiere `Authorization: Bearer <AI_SERVICE_API_KEY>`;
- `POST /v1/generation-requests/{requestId}/dispatch`: verifica una solicitud creada por backend, la publica en Vercel Queues y responde `202`.

## Vercel

Importar este repositorio como un proyecto FastAPI sin Build Command ni Output Directory. Usar `test` como Preview estable y `main` como Production. Configurar las mismas variables de `.env.example`, con valores y credenciales diferentes por ambiente. `DATABASE_URL` usa el rol runtime restringido de IA, nunca el rol migrador.

La cola y el consumidor se generan desde `vercel-queue`; la región queda en `gru1` y la concurrencia se limita a uno para no saturar Ollama. `LLM_API_URL` apunta al endpoint publicado mediante Cloudflare Tunnel y `LLM_API_TOKEN` autentica cada llamada como Bearer.

## Verificación

```bash
python -m ruff check .
python -m mypy src
python -m pytest
```

La interfaz con el backend está descrita en `architecture/data-interface.md` del [repositorio documental](https://github.com/Maico-Zurbriggen/proyecto-gimnasio-documentacion). Allí también viven el corpus funcional, la arquitectura y las reglas de dominio. Para trabajo asistido por IA, comenzar por su `AGENTS.md` y `manifest.json`.
