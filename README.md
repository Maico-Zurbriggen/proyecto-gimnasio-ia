# Proyecto Gimnasio — Servicio de IA y analítica

Servicio Python de generación online y procesos batch de análisis y machine learning.

## Responsabilidad

### Generación online

- exponer un OpenAPI versionado para backend;
- aceptar solicitudes asíncronas e idempotentes;
- orquestar el LLM configurado para el ambiente;
- persistir estados y resultados en estructuras autorizadas de PostgreSQL;
- registrar modelo, configuración, contrato e instante;
- no crear ni aprobar rutinas.

### Analítica batch

- construir features point-in-time;
- entrenar y evaluar contra criterios simples;
- calcular resultados reproducibles;
- escribir salidas precalculadas para backend.

## Despliegue objetivo

API FastAPI y consumidor durable se despliegan en Vercel. Vercel Queues desacopla la aceptación `202` del trabajo de generación. El LLM permanece en el Polo detrás de su API autenticada por ngrok, bajo el prefijo `/polo`. `LLM_API_TOKEN` contiene `POLO_API_TOKEN` y cada llamada envía `ngrok-skip-browser-warning: 1`; Ollama no se expone sin autenticación.

## Requisitos actuales

- Python 3.13;
- PostgreSQL local con el rol restringido de IA, o la base del deployment;
- acceso al endpoint autenticado del LLM del Polo para integración real.

## Inicio local

```bash
python -m venv .venv
# Activar el entorno virtual
python -m pip install -e ".[analytics,dev]"
cp .env.example .env
python dev_server.py
```

En PowerShell, usar `Copy-Item .env.example .env`. En Windows, el lanzador local fuerza el event loop Selector porque Uvicorn usa Proactor por defecto y Psycopg async requiere Selector. Backend es dueño de las migraciones; este repositorio no ejecuta cambios de esquema.

Para correr la generación local, establecer `APP_ENV=local` y `GENERATION_QUEUE_MODE=local` en `.env.local`. `DATABASE_URL` debe ser `postgresql://gym_ai_local@127.0.0.1:55432/gym_local`, sin el parámetro `schema` de Prisma; Backend crea la base y el rol restringido. `APP_ENV=local` rechaza bases remotas. El worker del servicio consulta las solicitudes persistidas cada dos segundos y se despierta al recibir un dispatch. Recupera pendientes sin intento y leases vencidos, procesa de a una y conserva el máximo de dos intentos tras reinicios. El proceso debe permanecer activo para trabajar; este modo sólo se permite en desarrollo.

El LLM sigue alojado en el Polo. El conector limita el contexto a 8192 tokens y `/ready` comprueba que el modelo configurado esté instalado. El modo local cambia PostgreSQL y el procesamiento de solicitudes; conserva la inferencia real por la API autenticada del Polo y las validaciones del backend.

La respuesta del Polo se recibe por streaming JSONL y sólo se acepta después de `done: true`. El intento completo conserva el límite de 120 segundos, aunque sigan llegando fragmentos. Antes de registrar `COMPLETADA`, IA comprueba catálogo, disponibilidad y las restricciones de prescripción entregadas por backend; una salida inválida usa el único reintento previsto. Las conexiones PostgreSQL vencen a los cinco segundos para permitir que el worker se recupere cuando la base vuelve a estar disponible.

Endpoints:

- `GET /health`: salud del proceso, público y sin consultar dependencias;
- `GET /ready`: verifica PostgreSQL y el LLM; requiere `Authorization: Bearer <AI_SERVICE_API_KEY>`;
- `POST /v1/generation-requests/{requestId}/dispatch`: verifica una solicitud creada por backend y responde `202`; usa Vercel Queues en despliegues y el worker local cuando `GENERATION_QUEUE_MODE=local`.

## Vercel

Importar este repositorio como un proyecto FastAPI sin Build Command ni Output Directory. Usar `test` como Preview estable y `main` como Production. Configurar las mismas variables de `.env.example`, con valores y credenciales diferentes por ambiente. `DATABASE_URL` usa el rol runtime restringido de IA, nunca el rol migrador.

La cola y el consumidor se generan desde `vercel-queue`; la región queda en `gru1` y la concurrencia se limita a uno para no saturar Ollama. `LLM_API_URL` apunta a `https://yen-entrench-grader.ngrok-free.dev/polo`; el conector consulta `/api/tags` y `/api/chat` con el Bearer de `POLO_API_TOKEN` y el encabezado `ngrok-skip-browser-warning: 1`. La configuración `generative/generar-rutina@10` usa un JSON Schema compatible como `format` y `think: false`, índices de catálogo y grupos de series con campos breves para reducir la inferencia. Su formato privado, compatibilidad con el runtime y expansión al contrato público están definidos en [la arquitectura generativa](https://github.com/Maico-Zurbriggen/proyecto-gimnasio-documentacion/blob/develop/architecture/generative-ai.md). El prompt recibe los rangos y cobertura que valida backend mediante `prescription_constraints`, las cantidades explícitas por día en `muscle_counts_per_day` y los músculos primarios del catálogo. La validación comprueba IDs distintos por músculo en cada día y no cuenta participación secundaria; un incumplimiento usa el reintento y nunca se completa como válido. Las llamadas locales a Vercel Queues requieren vincular el proyecto con `vercel link` y cargar sus variables con `vercel env pull`.

## Verificación

```bash
python -m ruff check .
python -m mypy src
python -m pytest
```

La interfaz con el backend está descrita en `architecture/data-interface.md` del [repositorio documental](https://github.com/Maico-Zurbriggen/proyecto-gimnasio-documentacion). Allí también viven el corpus funcional, la arquitectura y las reglas de dominio. Para trabajo asistido por IA, comenzar por su `AGENTS.md` y `manifest.json`.
