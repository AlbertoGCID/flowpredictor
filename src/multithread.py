from multiprocessing import Pool, cpu_count 
from typing import Dict, Any

def paralelizado(funcion: callable, datosprueba: Any) -> Dict[str, Any]:
    """
    Execute the given function in parallel using multiple CPU cores.

    Parameters:
    - funcion (callable): The function to be executed in parallel.
    - datosprueba (Any): Data to be passed to the function.

    Returns:
    - Dict[str, Any]: A dictionary containing the results of the parallel executions.
    """
    num_cores = cpu_count()
    print(f'This machine has {num_cores} cores.')
    
    with Pool(num_cores) as pool:
        margen = range(0, 200, 5)
        inputs = [(m, datosprueba) for m in margen]
        results = pool.map(funcion, inputs)

    return dict(zip(map(str, margen), results))





if __name__ == "__main__":
  print("Todas las librerías son cargadas correctamente")