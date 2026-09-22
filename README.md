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

API FastAPI y worker se despliegan juntos en Vercel. La API publica un mensaje en Vercel Queues
(request_id) al crear cada solicitud; `gym_engine.worker.queue_consumer` corre como consumer
push generado a partir de `[[tool.vercel.subscribers]]` en `pyproject.toml` y reclama esa fila
puntual con lease sobre PostgreSQL (`FOR UPDATE SKIP LOCKED`, igual que siempre) — Vercel Queues
es sólo el disparador, Postgres sigue siendo la única fuente de verdad de intentos/lease. El
reintento único (`GENERATION_MAX_RETRIES`) lo decide el consumer, no el redelivery nativo de
Vercel. El LLM permanece en el Polo detrás de un dominio HTTPS estable de Cloudflare Tunnel
protegido con un token Bearer; Ollama no se expone sin autenticación. También puede usarse OpenAI
como proveedor alternativo (`LLM_PROVIDER=openai`).

## Requisitos actuales

- Python 3.13;
- acceso autorizado a Neon Test con el rol restringido del servicio de IA;
- acceso al endpoint autenticado del LLM del Polo (o credenciales de OpenAI) para integración real.

## Inicio local

```bash
python -m venv .venv
# Activar el entorno virtual
python -m pip install -e ".[dev]"
cp .env.example .env
# En PowerShell: Copy-Item .env.example .env

# API (recibe la solicitud, 202 inmediato, publica el request_id en Vercel Queues)
python -m uvicorn gym_engine.api.app:app --reload

# Worker (procesa fuera de la petición HTTP, proceso separado). Fuera de Vercel corre en
# poll mode contra la cola real -- requiere el proyecto vinculado (`vercel link`) para que
# el SDK resuelva un token OIDC, o VERCEL_QUEUE_TOKEN seteado a mano.
python -m gym_engine.worker.poller
```

Backend es dueño de las migraciones; este repositorio no ejecuta cambios de esquema — las tablas
`ai_generation_*` deben existir de antemano en `DATABASE_URL`.

Endpoints:

- `GET /health`: salud del proceso, público y sin consultar dependencias;
- `GET /ready`: verifica PostgreSQL; requiere autenticación (`X-API-Key` o
  `Authorization: Bearer <AI_SERVICE_API_KEY>`);
- `POST /v1/routine-generations`: crea (o recupera, si el `idempotency_key` ya existe) una
  solicitud de generación de rutina y responde `202`/`200`;
- `GET /v1/routine-generations/{request_id}`: consulta el estado y, si está completa, la
  estructura candidata.

Ambos endpoints de negocio requieren `X-API-Key` o `Authorization: Bearer` con el mismo secreto
(`AI_SERVICE_API_KEY`) — se aceptan los dos esquemas mientras conviven clientes que usan uno u
otro.

## Vercel

Importar este repositorio como un proyecto FastAPI sin Build Command ni Output Directory. Usar
`test` como Preview estable y `main` como Production. Configurar las mismas variables de
`.env.example`, con valores y credenciales diferentes por ambiente. `DATABASE_URL` usa el rol
runtime restringido de IA, nunca el rol migrador. `LLM_API_URL` apunta al endpoint publicado
mediante Cloudflare Tunnel y `LLM_API_TOKEN` autentica cada llamada como Bearer.

Habilitar Vercel Queues (beta pública) en el proyecto: `pyproject.toml` ya declara el consumer
(`[[tool.vercel.subscribers]]`), así que Vercel lo genera como función privada air-gapped al
desplegar, sin tocar `vercel.json`.

## Verificación

```bash
python -m ruff check .
python -m mypy src
python -m pytest
```

La interfaz con el backend está descrita en `architecture/data-interface.md` del
[repositorio documental](https://github.com/Maico-Zurbriggen/proyecto-gimnasio-documentacion). Allí
también viven el corpus funcional, la arquitectura y las reglas de dominio. Para trabajo asistido
por IA, comenzar por su `AGENTS.md` y `manifest.json`.
