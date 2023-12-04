# TO-DO

- preliminar creación de la clase para trabajar
1. Integración de las diferentes fuentes
    a. Lectura de la fuente a un csv (función clase add csv, lee todas las columnas como fuentes de diferentes zonas)
    b. Integración a un dataframe
        i. Asignación de los nombres de las columnas con la funcion asignación de nombres
        ii. Asignación de la columna de salida
        iii. verificación de la columna temporal
2. Normalización del dataset y preparación de las ventanas temporales
    a. Test de normalización
    b. Test de ventana temporal
3. Selección de función de ejecución (entrenamiento, predicción)
    a. Entrenamiento, debe fijar split, renormalizar, partir dataframe, devolver valores de ns, R2 y f1 score
    b. predicción por fecha, indicas fecha a predecir y te devuelve predicciones de hasta 5 días (tiene 5 modelos entrenados)
4. Predicción a mano: pide los datos por teclado o por csv y devuelve la predicción de caudal para esos datos


# 1. Configuración del Entorno Local

## 1.1 Instalar Python 3.9

Descarga e instala Python 3.9 desde el [sitio web oficial de Python](https://www.python.org/downloads/).

## 1.2 Instalar Virtualenv

```bash
pip install virtualenv

virtualenv venv

venv\Scripts\activate

source venv/bin/activate
```


# 2. Estructura de Carpetas y Archivos
## 2.1 Estructura Básica

```
/your_project
|-- /src
|   |-- __init__.py
|   |-- main.py
|-- /tests
|   |-- __init__.py
|   |-- test_main.py
|-- README.md
|-- requirements.txt
|-- .gitignore
```

## 2.2 Archivos de Configuración

- README.md: Documentación principal del proyecto.
- requirements.txt: Lista de dependencias del proyecto.


# 3. Configuración de Git
## 3.1 Inicializar un Repositorio Git

```git init```

## 3.2 Crear un archivo .gitignore

```
# .gitignore
__pycache__/
venv/
*.pyc
*.pyo
*.pyd
```

# 4. Desarrollo del Proyecto
## 4.1 Desarrollar el Código Principal
En `src/main.py`, puedes comenzar a desarrollar tu código.

## 4.2 Documentar el Código
Asegúrate de agregar documentación adecuada en tu código y README.

# 5. Configuración de Pruebas
## 5.1 Utilizar Unittest
Escribe pruebas unitarias en tests/test_main.py utilizando el módulo unittest.

# 6. Empaquetado y Distribución
## 6.1 Crear un archivo setup.py

```python
# setup.py
from setuptools import setup, find_packages

setup(
    name='your_project',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # Lista de tus dependencias
    ],
)
```

## 6.2 Crear un archivo MANIFEST.in

```
# MANIFEST.in
include README.md
```

# 7. Publicar en GitHub
## 7.1 Crear un Repositorio en GitHub
Crea un nuevo repositorio en GitHub.
## 7.2 Enlazar el Repositorio Local con GitHub
```
git remote add origin <URL del repositorio en GitHub>
```

## 7.3 Hacer el Primer Commit y Push

```
git add .
git commit -m "Primer commit"
git push -u origin master
```


# 8. Configuración de CI/CD (Opcional)
Configura servicios de CI/CD como GitHub Actions o Travis CI para ejecutar pruebas automáticamente.

# 9. Documentación Adicional (Opcional)
Añade documentación adicional, como instrucciones de instalación y uso, en tu README.

# 10. Liberación de Versiones (Opcional)
Considera el uso de herramientas como bump2version para gestionar versiones de manera automática.

Con estos pasos, deberías tener un proyecto básico de IA en Python listo para ser compartido y desarrollado en cualquier equipo. ¡Buena suerte con tu proyecto!