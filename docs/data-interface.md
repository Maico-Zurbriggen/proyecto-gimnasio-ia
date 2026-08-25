# Interfaz de datos con el backend

## Principio

PostgreSQL es la frontera operativa inicial, pero el motor no depende de tablas internas sin un acuerdo explícito. Cada job declara:

- nombre y versión del dataset de entrada;
- columnas, tipos, nulabilidad y unidades;
- instante de corte y zona horaria;
- claves de idempotencia;
- tabla o artefacto de salida;
- versión del componente y parámetros.

## Desarrollo local

Cada integrante configura `DATABASE_URL` en `.env`. Se prefieren snapshots y seeds sintéticos para tests reproducibles. Nunca se copian datos personales reales al repositorio ni a notebooks.

## Cambios

Un cambio incompatible requiere:

1. documentación de la nueva versión;
2. migración o vista preparada por el backend;
3. compatibilidad temporal cuando sea necesaria;
4. PR relacionados en backend e IA;
5. actualización de fixtures y tests de contrato.

El motor sólo escribe en estructuras de salida designadas. No actualiza sesiones, rutinas, usuarios ni otras fuentes transaccionales.
