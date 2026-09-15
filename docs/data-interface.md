# Contrato de API — servicio de IA (backend → servicio IA)

| | |
|---|---|
| **Versión** | v1 |
| **Base path** | `/v1/routine-generations` |
| **Alcance** | Generación de rutina (FL-04), subconjunto sin HITL. Ver `maquina-estados-langgraph.pdf` para el diseño completo y qué falta. |
| **Auth** | Header `X-API-Key`, obligatorio en todos los endpoints. Ver [Autenticación](#autenticación). |
| **Formato** | JSON. `Content-Type: application/json` en request y response. |

Este documento describe únicamente el contrato HTTP. El diseño interno (grafo LangGraph, worker, persistencia) está en `maquina-estados-langgraph.pdf` y no es parte de este contrato.

---

## Autenticación

Todos los endpoints bajo `/v1/routine-generations` exigen el header:

```
X-API-Key: <clave>
```

La clave esperada la define el servicio de IA por variable de entorno (`AI_SERVICE_API_KEY_TEST` o `AI_SERVICE_API_KEY_PRODUCTION` según `APP_ENV`) — el backend la manda tal cual, no hay negociación ni token con expiración.

| Situación | Código |
|---|---|
| Header ausente | `401` |
| Header con clave incorrecta | `401` |
| Servicio sin ninguna clave configurada del lado de IA | `500` (falla cerrado, no deja pasar por defecto) |
| Header correcto | sigue al handler normalmente |

No hay scopes ni roles — es una sola clave compartida por ambiente, pensada para el canal backend→servicio IA, no para llamar directo desde el frontend.

---

## Regla de diseño que condiciona el contrato

El servicio de IA **no tiene acceso a las tablas de dominio** (`Ejercicio`, `GrupoMuscular`, perfil del alumno, etc. — esquema `app`, propiedad del backend). Por eso `catalogo_prefiltrado` y `contexto_minimizado` no los calcula este servicio: **los arma el backend** y viajan como input en la solicitud. Si el backend manda un catálogo vacío o un contexto incompleto, la generación falla — no hay fallback que la complete del lado del servicio de IA.

---

## `POST /v1/routine-generations`

Crea una solicitud de generación. **Asíncrono**: no espera al LLM, responde de inmediato. El procesamiento real lo hace un worker aparte; el resultado se consulta con el `GET` de abajo.

### Request body

`Content-Type: application/json`. Cualquier campo no listado acá es rechazado (`extra: forbid`).

| Campo | Tipo | Requerido | Notas |
|---|---|---|---|
| `idempotency_key` | `string` (min 1 char) | Sí | Repetir la misma clave no crea una solicitud nueva: devuelve la existente tal cual está. |
| `gym_id` | `UUID` | Sí | |
| `student_id` | `UUID` | Sí | |
| `requested_by_user_id` | `UUID` | Sí | |
| `texto_libre` | `string \| null` | No | Pedido en lenguaje natural. Si viene y `parametros` es `null`, el LLM lo interpreta primero (nodo `interpretar_solicitud`). |
| `parametros` | [`ParametrosRutina`](#parametrosrutina) `\| null` | No | Si ya los tenés confirmados (HITL 1 resuelto río arriba, fuera de este servicio), mandalos acá y te salteás la interpretación. |
| `catalogo_prefiltrado` | `EjercicioRef[]` (mínimo 1) | Sí | Subconjunto ya filtrado por compatibilidad e inventario (RN-44a-d, RN-45). El LLM sólo puede usar ids de esta lista. |
| `contexto_minimizado` | [`MinimizedContext`](#minimizedcontext) | Sí | Sin nombre, correo, teléfono ni documento. |

`texto_libre` y `parametros` no pueden ser ambos `null` — si faltan los dos, la solicitud se crea igual (`202`) pero el grafo la marca `failed` al procesarla (violación: "faltan parametros").

#### `EjercicioRef`

| Campo | Tipo |
|---|---|
| `id` | `UUID` |
| `nombre` | `string` |
| `patron_movimiento` | `string` |

#### `MinimizedContext`

| Campo | Tipo | Notas |
|---|---|---|
| `nivel_experiencia` | `string` | |
| `dias_semanales_disponibles` | `int` (1–7) | |
| `objetivos_activos` | `string[]` | default `[]` |
| `condiciones` | `string[]` | default `[]` |

#### `ParametrosRutina`

| Campo | Tipo | Notas |
|---|---|---|
| `objetivo` | `string` | Sin enum cerrado todavía — pendiente de D2. No inventar valores del lado del backend sin avisar. |
| `frecuencia_semanal` | `int` (1–7) | RN-38 |
| `duracion_minutos` | `int` (> 0) | |
| `restricciones` | `string[]` | default `[]` |
| `confianza` | `float` (0–1) | default `1.0` |

### Respuestas

| Código | Cuándo | Body |
|---|---|---|
| `202` | Solicitud nueva, creada | `{"request_id": UUID, "status": "pending"}` |
| `200` | `idempotency_key` ya existía | `{"request_id": UUID, "status": "<estado actual>"}` — no reprocesa nada |
| `422` | Body inválido (falta campo, `catalogo_prefiltrado` vacío, UUID mal formado, etc.) | Error estándar de validación de FastAPI |

### Ejemplo

```bash
curl -X POST http://localhost:8000/v1/routine-generations \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <clave>" \
  -d '{
    "idempotency_key": "req-001",
    "gym_id": "11111111-1111-1111-1111-111111111111",
    "student_id": "22222222-2222-2222-2222-222222222222",
    "requested_by_user_id": "33333333-3333-3333-3333-333333333333",
    "texto_libre": "quiero ganar fuerza, 3 dias por semana",
    "catalogo_prefiltrado": [
      {"id": "44444444-4444-4444-4444-444444444444", "nombre": "Sentadilla", "patron_movimiento": "squat"}
    ],
    "contexto_minimizado": {
      "nivel_experiencia": "intermedio",
      "dias_semanales_disponibles": 3
    }
  }'
```

```json
{ "request_id": "5b1e...", "status": "pending" }
```

---

## `GET /v1/routine-generations/{request_id}`

Consulta el estado y, si terminó, el resultado. No hay push/webhook — es polling del lado del backend.

### Respuestas

| Código | Cuándo | Body |
|---|---|---|
| `200` | `request_id` existe | ver abajo |
| `404` | `request_id` no existe | `{"detail": "request_id no encontrado"}` |

### Body (`200`)

| Campo | Tipo | Notas |
|---|---|---|
| `request_id` | `UUID` | |
| `status` | `"pending" \| "processing" \| "completed" \| "failed"` | |
| `estructura_candidata` | [`RutinaEstructurada`](#rutinaestructurada) `\| null` | Presente solo si `status == "completed"` |
| `violaciones` | `string[] \| null` | Motivo de rechazo de la última validación, si la hubo (aplica tanto a intentos fallidos con reintento como al fallo final) |
| `error` | `string \| null` | Presente solo si `status == "failed"` |

#### `RutinaEstructurada`

```
{
  "dias": [
    {
      "orden": 1,
      "nombre": "Dia 1",
      "ejercicios": [
        {
          "ejercicio_id": "<uuid, pertenece a catalogo_prefiltrado>",
          "orden": 1,
          "nota": "string | null",
          "series": [
            {
              "orden": 1,
              "repeticiones_min": 8,
              "repeticiones_max": 12,
              "carga_sugerida": "string | null",
              "descanso_segundos": 60,
              "es_calentamiento": false
            }
          ]
        }
      ]
    }
  ]
}
```

### Ejemplo

```bash
curl http://localhost:8000/v1/routine-generations/5b1e... \
  -H "X-API-Key: <clave>"
```

```json
{
  "request_id": "5b1e...",
  "status": "completed",
  "estructura_candidata": { "dias": [ ... ] },
  "violaciones": null,
  "error": null
}
```

---

## Fuera de este contrato (a propósito, por ahora)

No están implementados en esta versión — no asumir que existen del lado del backend:

- Confirmación de parámetros (HITL 1) y revisión del profesor (HITL 2): en esta corrida el servicio va directo de parámetros a generación, sin pausas.
- `PATCH`/`DELETE` sobre una solicitud, o endpoint de cancelación.
- Persistencia/activación de la rutina en el esquema `app` — este servicio nunca escribe ahí (AGENTS.md); eso lo hace el backend con el resultado de `GET`.
- Notificación push cuando termina — solo polling.
- Scopes/roles en la API key — es una sola clave compartida por ambiente, sin expiración ni rotación automática.

## Notas operativas

- **Reintentos**: el servicio reintenta 1 vez la generación ante salida inválida o LLM caído (`GENERATION_MAX_RETRIES`), sin exponer eso en la API — el backend solo ve el resultado final.
- **Falla del worker a mitad de proceso**: una solicitud puede quedar en `processing` indefinidamente si el proceso del worker cae en el medio (no hay lease/heartbeat todavía). Si el backend ve un `processing` colgado por mucho más que `GENERATION_TIMEOUT_SECONDS * 2`, es ese bug — no reintentar automáticamente con la misma `idempotency_key` porque va a devolver la misma fila colgada.
