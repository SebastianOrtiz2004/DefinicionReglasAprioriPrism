
import sys
import math
import random
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional

# Asegurar soporte de UTF-8 en la consola de Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


class AprendizInductivoPRISM:

    def __init__(self, ruta_archivo: str, columna_objetivo: Optional[str] = None, valor_positivo: Optional[str] = None, semilla: int = 42):

        self.ruta_archivo = ruta_archivo
        self.semilla = semilla
        random.seed(self.semilla)

        self.df = self._cargar_csv(ruta_archivo)
        self.columna_objetivo = columna_objetivo or self._detectar_columna_objetivo()
        self.valor_positivo = valor_positivo or self._detectar_valor_positivo()
        self.columnas_predictivas = self._identificar_columnas_predictivas()

        # Métricas globales iniciales
        self.N = len(self.df)
        self.mascara_B = (self.df[self.columna_objetivo].astype(str).str.strip().str.lower() == str(self.valor_positivo).strip().lower())
        self.conteo_B = int(self.mascara_B.sum())
        self.probabilidad_a_priori_B = self.conteo_B / self.N if self.N > 0 else 0.0

        # Historial de iteraciones
        self.historial = []

    def _cargar_csv(self, ruta_archivo: str) -> pd.DataFrame:
        """Carga el CSV probando distintas codificaciones comunes (UTF-8, Latin1, CP1252)."""
        codificaciones = ['utf-8', 'latin1', 'cp1252', 'utf-8-sig']
        for cod in codificaciones:
            try:
                df = pd.read_csv(ruta_archivo, encoding=cod)
                # Limpiar espacios en nombres de columnas
                df.columns = [col.strip() for col in df.columns]
                return df
            except (UnicodeDecodeError, Exception):
                continue
        raise ValueError(f"No se pudo cargar el archivo '{ruta_archivo}' con ninguna codificación probada.")

    def _detectar_columna_objetivo(self) -> str:
        """Detecta la columna clase buscando palabras clave o seleccionando la última columna."""
        palabras_clave = ['resultado', 'exito', 'exito_proyecto', 'jugar', 'clase', 'target', 'label', 'class']
        for col in self.df.columns:
            if col.lower() in palabras_clave:
                return col
        return self.df.columns[-1]

    def _detectar_valor_positivo(self) -> Any:
        """Detecta el valor positivo de la columna objetivo."""
        valores_unicos = self.df[self.columna_objetivo].dropna().unique()
        palabras_positivas = ['si', 'sí', 'yes', 'true', '1', 'exitoso', 'éxito', 'exito', 'positivo', '1.0']
        for val in valores_unicos:
            if str(val).strip().lower() in palabras_positivas:
                return val
        return valores_unicos[0]

    def _identificar_columnas_predictivas(self) -> List[str]:
        """Identifica atributos predictivos excluyendo identificadores y la columna objetivo."""
        palabras_id = ['id', 'proyecto_id', 'index', 'nro', 'codigo', 'cod']
        predictivas = []
        for col in self.df.columns:
            if col == self.columna_objetivo:
                continue
            if col.lower() in palabras_id or col.lower().endswith('_id'):
                continue
            predictivas.append(col)
        return predictivas

    def _evaluar_regla_candidata(self, condiciones_actuales: Dict[str, Any], columna: str, valor: Any) -> Dict[str, Any]:
        """
        Evalúa una regla candidata agregando (columna = valor) a las condiciones actuales.
        Calcula: |A|, Cobertura (|A ∩ B|), Confianza, Soporte y Lift.
        """
        condiciones_candidatas = dict(condiciones_actuales)
        condiciones_candidatas[columna] = valor

        # Crear máscara booleana para el antecedente A
        mascara_A = pd.Series(True, index=self.df.index)
        for c, v in condiciones_candidatas.items():
            mascara_A = mascara_A & (self.df[c] == v)

        casos_A = int(mascara_A.sum())
        cobertura = int((mascara_A & self.mascara_B).sum())

        confianza = cobertura / casos_A if casos_A > 0 else 0.0
        soporte = cobertura / self.N if self.N > 0 else 0.0
        lift = confianza / self.probabilidad_a_priori_B if (self.probabilidad_a_priori_B > 0 and confianza > 0) else 0.0

        # Construir representación legible completa de la regla
        terminos_regla = [f"{c} = '{v}'" for c, v in condiciones_candidatas.items()]
        antecedente = " AND ".join(terminos_regla)
        texto_regla = f"SI {antecedente} ENTONCES {self.columna_objetivo} = '{self.valor_positivo}'"

        return {
            'columna_agregada': columna,
            'valor_agregado': valor,
            'condiciones': condiciones_candidatas,
            'antecedente': antecedente,
            'texto_regla': texto_regla,
            'casos_A': casos_A,
            'cobertura': cobertura,
            'confianza': confianza,
            'soporte': soporte,
            'lift': lift
        }

    def _seleccionar_regla_ganadora(self, candidatas: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], bool, int]:
        """
        Aplica la jerarquía formal de resolución de empates:
        1. Mayor Confianza.
        2. Mayor Cobertura.
        3. Mayor Lift.
        4. Selección Aleatoria si persiste empate absoluto.
        """
        def funcion_ordenamiento(c):
            return (round(c['confianza'], 8), c['cobertura'], round(c['lift'], 8))

        candidatas_ordenadas = sorted(candidatas, key=funcion_ordenamiento, reverse=True)
        mejor_puntaje = funcion_ordenamiento(candidatas_ordenadas[0])

        # Filtrar candidatas en empate perfecto en las 3 métricas
        empates_top = [c for c in candidatas_ordenadas if funcion_ordenamiento(c) == mejor_puntaje]

        hubo_empate = len(empates_top) > 1
        if hubo_empate:
            regla_ganadora = random.choice(empates_top)
        else:
            regla_ganadora = candidatas_ordenadas[0]

        return regla_ganadora, hubo_empate, len(empates_top)

    def ejecutar(self) -> str:
        """
        Ejecuta el proceso iterativo PRISM y genera el informe completo formateado en español.
        """
        lineas_salida = []

        encabezado = "=" * 85 + "\n"
        encabezado += "           INFORME DE EJECUCIÓN: ALGORITMO PRISM ADAPTADO (ACADÉMICO)\n"
        encabezado += "=" * 85 + "\n"
        encabezado += f" Archivo de datos:             '{self.ruta_archivo}'\n"
        encabezado += f" Total de Observaciones (N):   {self.N}\n"
        encabezado += f" Columna Objetivo (Clase):     '{self.columna_objetivo}'\n"
        encabezado += f" Valor Positivo de la Clase:   '{self.valor_positivo}'\n"
        encabezado += f" Conteo Clase Positiva (|B|):  {self.conteo_B}\n"
        encabezado += f" Proporción / Probabilidad Base P(B): {self.probabilidad_a_priori_B:.6f} ({self.probabilidad_a_priori_B * 100:.2f}%)\n"
        encabezado += f" Atributos Predictivos ({len(self.columnas_predictivas)}): {', '.join(self.columnas_predictivas)}\n"
        encabezado += "=" * 85 + "\n"

        lineas_salida.append(encabezado)
        print(encabezado)

        condiciones_actuales = {}
        columnas_restantes = list(self.columnas_predictivas)

        # Bucle iterativo desde 1 hasta K (número total de atributos predictivos)
        for iteracion in range(1, len(self.columnas_predictivas) + 1):
            candidatas = []

            # Generar y evaluar candidatas con los atributos restantes
            for col in columnas_restantes:
                valores_unicos = self.df[col].dropna().unique()
                for val in valores_unicos:
                    cand = self._evaluar_regla_candidata(condiciones_actuales, col, val)
                    candidatas.append(cand)

            # Ordenar candidatas según la jerarquía para visualización
            candidatas_ordenadas = sorted(candidatas, key=lambda c: (
                round(c['confianza'], 8), 
                c['cobertura'], 
                round(c['lift'], 8)
            ), reverse=True)

            # Seleccionar regla ganadora aplicando jerarquía de empates
            regla_ganadora, hubo_empate, numero_empates = self._seleccionar_regla_ganadora(candidatas)

            # Construir Tabla ASCII formateada para la iteración actual
            max_longitud_regla = max(len(c['texto_regla']) for c in candidatas_ordenadas)
            ancho_col_regla = max(55, max_longitud_regla + 2)
            ancho_tabla_total = ancho_col_regla + 65

            iter_encabezado = f"\n>>> ITERACIÓN {iteracion} / {len(self.columnas_predictivas)} (Evaluando {len(candidatas)} Reglas Candidatas)\n"
            iter_encabezado += "-" * ancho_tabla_total + "\n"
            iter_encabezado += f"{'#':<3} | {'Regla Candidata Completa':<{ancho_col_regla}} | {'|A|':<6} | {'|A&B|':<6} | {'Confianza':<10} | {'Soporte':<10} | {'Lift':<8} |\n"
            iter_encabezado += "-" * ancho_tabla_total + "\n"

            iter_cuerpo = ""
            for idx, c in enumerate(candidatas_ordenadas, 1):
                marca_ganadora = " [*]" if c == regla_ganadora else "    "
                iter_cuerpo += f"{idx:<3}{marca_ganadora}| {c['texto_regla']:<{ancho_col_regla}} | {c['casos_A']:<6} | {c['cobertura']:<6} | {c['confianza']:<10.6f} | {c['soporte']:<10.6f} | {c['lift']:<8.4f} |\n"

            iter_pie = "-" * ancho_tabla_total + "\n"
            if hubo_empate:
                iter_pie += f" NOTA: ¡Empate técnico detectado entre {numero_empates} reglas top! Se aplicó desempate aleatorio (semilla={self.semilla}).\n"

            anuncio_ganadora = (
                f"\n  ===================================================================================================\n"
                f"  LA REGLA QUE PASA ES (Iteración {iteracion}):\n"
                f"  {regla_ganadora['texto_regla']}\n"
                f"  -> Confianza: {regla_ganadora['confianza']:.6f} | Cobertura (|A & B|): {regla_ganadora['cobertura']} | Soporte: {regla_ganadora['soporte']:.6f} | Lift: {regla_ganadora['lift']:.4f}\n"
                f"  ===================================================================================================\n"
            )

            texto_iteracion_completo = iter_encabezado + iter_cuerpo + iter_pie + anuncio_ganadora
            lineas_salida.append(texto_iteracion_completo)
            print(texto_iteracion_completo)

            # Actualizar condiciones ganadoras y remover atributo usado
            condiciones_actuales[regla_ganadora['columna_agregada']] = regla_ganadora['valor_agregado']
            columnas_restantes.remove(regla_ganadora['columna_agregada'])

            # Guardar en historial
            self.historial.append({
                'iteracion': iteracion,
                'regla_ganadora': regla_ganadora,
                'numero_candidatas': len(candidatas)
            })

        # Resumen final de la Regla Inducida Progresiva
        resumen = "\n" + "=" * 85 + "\n"
        resumen += "                    RESUMEN FINAL DE LA REGLA INDUCIDA\n"
        resumen += "=" * 85 + "\n"
        ganadora_final = self.historial[-1]['regla_ganadora']
        resumen += f" REGLA COMPLETA GANADORA:\n"
        resumen += f"   {ganadora_final['texto_regla']}\n\n"
        resumen += f" MÉTRICAS FINALES:\n"
        resumen += f"   - Casos A (|A|):                   {ganadora_final['casos_A']}\n"
        resumen += f"   - Cobertura (|A ∩ B|):              {ganadora_final['cobertura']}\n"
        resumen += f"   - Confianza (Precisión Local):     {ganadora_final['confianza']:.6f} ({ganadora_final['confianza']*100:.2f}%)\n"
        resumen += f"   - Soporte (Proporción Global N):   {ganadora_final['soporte']:.6f} ({ganadora_final['soporte']*100:.2f}%)\n"
        resumen += f"   - Lift (Fuerza de Correlación):    {ganadora_final['lift']:.4f}\n"
        resumen += "=" * 85 + "\n"

        lineas_salida.append(resumen)
        print(resumen)

        reporte_completo = "".join(lineas_salida)
        return reporte_completo


if __name__ == "__main__":
    archivo_csv = "dataset_prism_proyectos_software.csv"
    aprendiz = AprendizInductivoPRISM(
        ruta_archivo=archivo_csv,
        columna_objetivo="Exito_Proyecto",
        valor_positivo="Si",
        semilla=42
    )
    aprendiz.ejecutar()
