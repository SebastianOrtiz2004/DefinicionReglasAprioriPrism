import sys
import math
import pandas as pd
from typing import List, Dict, Set, Tuple, Any, Optional

# Asegurar soporte de UTF-8 en la consola de Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


class ClasificadorApriori:
    """
    Clase principal en español para la ejecución formal del Algoritmo APRIORI.
    """

    def __init__(self, ruta_archivo: str, soporte_minimo_pct: float = 0.70, confianza_minima_base: float = 0.85, confianza_estricta: float = 0.90):
        """
        Inicializa el clasificador APRIORI con los hiperparámetros.
        """
        self.ruta_archivo = ruta_archivo
        self.soporte_minimo_pct = soporte_minimo_pct
        self.confianza_minima_base = confianza_minima_base
        self.confianza_estricta = confianza_estricta

        self.df = self._cargar_csv(ruta_archivo)
        self.N = len(self.df)
        self.cobertura_minima = math.ceil(self.N * self.soporte_minimo_pct)

        # Identificar columnas de atributos excluyendo identificadores
        self.columnas_atributos = self._identificar_columnas_atributos()

        # Diccionario para almacenar la cobertura de todas las series de atributos
        self.series_atributos = self._construir_series_atributos()
        self.diccionario_coberturas = {}  # frozenset -> cobertura

        # Estructura para almacenar ítem-sets frecuentes por nivel k
        self.itemsets_frecuentes_por_nivel = {}  # k -> dict(frozenset -> cobertura)
        self.reporte_fase1 = {}  # k -> lista de diccionarios

        # Estructura para almacenar reglas evaluadas
        self.reglas_evaluadas = []

    def _cargar_csv(self, ruta_archivo: str) -> pd.DataFrame:
        """Carga el CSV probando distintas codificaciones comunes (UTF-8, Latin1, CP1252)."""
        codificaciones = ['utf-8', 'latin1', 'cp1252', 'utf-8-sig']
        for cod in codificaciones:
            try:
                df = pd.read_csv(ruta_archivo, encoding=cod)
                df.columns = [col.strip() for col in df.columns]
                return df
            except (UnicodeDecodeError, Exception):
                continue
        raise ValueError(f"No se pudo cargar el archivo '{ruta_archivo}' con ninguna codificación probada.")

    def _identificar_columnas_atributos(self) -> List[str]:
        """Identifica columnas predictivas excluyendo identificadores de transacción."""
        palabras_id = ['id', 'transaccion_id', 'transaccion', 'index', 'nro']
        columnas = []
        for col in self.df.columns:
            if col.lower() in palabras_id or col.lower().endswith('_id'):
                continue
            columnas.append(col)
        return columnas

    def _construir_series_atributos(self) -> Dict[str, pd.Series]:
        """Crea filtros booleanos para cada par atributo=valor ordenados consecutivamente (ej. 'Leche=1', 'Leche=0')."""
        series_dict = {}
        for col in self.columnas_atributos:
            for val in [1, 0]:
                texto_item = f"{col}={val}"
                series_dict[texto_item] = (self.df[col] == val)
        return series_dict

    def _calcular_cobertura_itemset(self, conjunto_items: frozenset) -> int:
        """Calcula la cobertura conjunta absoluta (frecuencia) de un ítem-set en la base de datos."""
        if conjunto_items in self.diccionario_coberturas:
            return self.diccionario_coberturas[conjunto_items]

        filtro_conjunto = pd.Series(True, index=self.df.index)
        for item in conjunto_items:
            filtro_conjunto = filtro_conjunto & self.series_atributos[item]

        cobertura = int(filtro_conjunto.sum())
        self.diccionario_coberturas[conjunto_items] = cobertura
        return cobertura

    def _generar_combinaciones(self, elementos: list, r: int) -> List[tuple]:
        """Genera combinaciones de tamaño r a partir de una lista sin usar librerías externas."""
        if r == 0:
            return [()]
        if not elementos:
            return []
        primer_elem = elementos[0]
        resto = elementos[1:]
        con_primer = [(primer_elem,) + combo for combo in self._generar_combinaciones(resto, r - 1)]
        sin_primer = self._generar_combinaciones(resto, r)
        return con_primer + sin_primer

    def generar_encabezado(self) -> str:
        """Genera el encabezado inicial con los datos generales del problema."""
        texto = "=" * 90 + "\n"
        texto += "              INFORME DE EJECUCIÓN: ALGORITMO APRIORI DE ASOCIACIÓN\n"
        texto += "=" * 90 + "\n"
        texto += f" Archivo cargado:                    '{self.ruta_archivo}'\n"
        texto += f" Total de Transacciones (N):          {self.N}\n"
        texto += f" Soporte Mínimo:                     {self.soporte_minimo_pct:.2f}\n"
        texto += f" Confianza Mínima Base:              {self.confianza_minima_base:.2f}\n"
        texto += f" Ítems Atributo=Valor Disponibles:    {len(self.series_atributos)} pares ({len(self.columnas_atributos)} atributos x 2 valores)\n"
        texto += "=" * 90 + "\n"
        return texto

    def ejecutar_fase0(self) -> str:
        """FASE 0: Cobertura Mínima."""
        texto = "\n" + "=" * 90 + "\n"
        texto += "                         FASE 0: COBERTURA MÍNIMA\n"
        texto += "=" * 90 + "\n"
        texto += f" Cobertura Mínima: {self.N} * {self.soporte_minimo_pct:.2f} = {self.cobertura_minima} transacciones\n"
        texto += "=" * 90 + "\n"
        return texto

    def ejecutar_fase1(self) -> str:
        """FASE 1: Generación y Poda de Ítem-sets Frecuentes (k = 1, 2, 3, ...)."""
        salida_fase1 = []

        # ---------------------------------------------------------------------
        # Nivel k = 1 (Agrupado consecutivamente: Atributo=1 seguido de Atributo=0)
        # ---------------------------------------------------------------------
        k = 1
        itemsets_frecuentes_k1 = {}
        reporte_k1 = []

        for texto_item in self.series_atributos.keys():
            conjunto_items = frozenset([texto_item])
            cobertura = self._calcular_cobertura_itemset(conjunto_items)
            aprobado = cobertura >= self.cobertura_minima

            reporte_k1.append({
                'conjunto_items': conjunto_items,
                'texto_itemset': f"{{{texto_item}}}",
                'cobertura': cobertura,
                'soporte': cobertura / self.N,
                'estado': "APROBADO" if aprobado else "DESCARTADO"
            })

            if aprobado:
                itemsets_frecuentes_k1[conjunto_items] = cobertura

        self.itemsets_frecuentes_por_nivel[1] = itemsets_frecuentes_k1
        # Mantener el orden de atributos agrupados (Pan=1, Pan=0, Leche=1, Leche=0, ...)
        self.reporte_fase1[1] = reporte_k1

        # ---------------------------------------------------------------------
        # Niveles k >= 2 (Bucle Iterativo de Apriori hasta incluir k=3)
        # ---------------------------------------------------------------------
        k = 2
        items_aprobados_k1 = list(itemsets_frecuentes_k1.keys())

        while k <= 3:
            candidatos_k = set()
            
            if k == 2:
                frecuentes_previos = list(self.itemsets_frecuentes_por_nivel[1].keys())
                for i in range(len(frecuentes_previos)):
                    for j in range(i + 1, len(frecuentes_previos)):
                        conjunto1 = frecuentes_previos[i]
                        conjunto2 = frecuentes_previos[j]
                        conjunto_union = conjunto1 | conjunto2

                        if len(conjunto_union) == 2:
                            atributos = [item.split('=')[0] for item in conjunto_union]
                            if len(atributos) == len(set(atributos)):
                                candidatos_k.add(conjunto_union)

            elif k == 3:
                # Generar candidatos de tamaño 3 a partir de los ítems aprobados de k=1
                frecuentes_previos = items_aprobados_k1
                for combo in self._generar_combinaciones(frecuentes_previos, 3):
                    conjunto_union = combo[0] | combo[1] | combo[2]
                    if len(conjunto_union) == 3:
                        atributos = [item.split('=')[0] for item in conjunto_union]
                        if len(atributos) == len(set(atributos)):
                            candidatos_k.add(conjunto_union)

            if not candidatos_k:
                break

            # Evaluación de Cobertura de candidatos válidos de nivel k
            itemsets_frecuentes_k = {}
            reporte_k = []

            for candidato in candidatos_k:
                cobertura = self._calcular_cobertura_itemset(candidato)
                aprobado = cobertura >= self.cobertura_minima

                texto_items = " AND ".join(sorted(list(candidato)))
                reporte_k.append({
                    'conjunto_items': candidato,
                    'texto_itemset': f"{{{texto_items}}}",
                    'cobertura': cobertura,
                    'soporte': cobertura / self.N,
                    'estado': "APROBADO" if aprobado else "DESCARTADO"
                })

                if aprobado:
                    itemsets_frecuentes_k[candidato] = cobertura

            self.reporte_fase1[k] = sorted(reporte_k, key=lambda x: x['cobertura'], reverse=True)

            if itemsets_frecuentes_k:
                self.itemsets_frecuentes_por_nivel[k] = itemsets_frecuentes_k
            else:
                self.itemsets_frecuentes_por_nivel[k] = {}

            k += 1

        # Formatear tablas ASCII para cada nivel k de la Fase 1
        texto = "\n" + "=" * 90 + "\n"
        texto += "         FASE 1: GENERACIÓN Y PODA DE ÍTEM-SETS FRECUENTES (k = 1, 2, 3)\n"
        texto += "=" * 90 + "\n"

        for nivel in sorted(self.reporte_fase1.keys()):
            reporte_nivel = self.reporte_fase1[nivel]
            frecuentes_cont = sum(1 for item in reporte_nivel if item['estado'] == "APROBADO")
            descartados_cont = len(reporte_nivel) - frecuentes_cont

            longitud_maxima = max(len(item['texto_itemset']) for item in reporte_nivel)
            ancho_columna = max(50, longitud_maxima + 2)
            ancho_tabla = ancho_columna + 45

            encabezado_tabla = f"\n>>> ÍTEM-SETS CANDIDATOS DE TAMAÑO k = {nivel} (Total: {len(reporte_nivel)} | Aprobados: {frecuentes_cont} | Descartados: {descartados_cont})\n"
            encabezado_tabla += "-" * ancho_tabla + "\n"
            encabezado_tabla += f"{'#':<3} | {'Ítem-set (Atributos = Valor)':<{ancho_columna}} | {'Cobertura':<10} | {'Soporte':<10} | {'Estado (>= ' + str(self.cobertura_minima) + ')':<15} |\n"
            encabezado_tabla += "-" * ancho_tabla + "\n"

            cuerpo = ""
            for idx, r in enumerate(reporte_nivel, 1):
                marca = " [X]" if r['estado'] == "APROBADO" else "    "
                cuerpo += f"{idx:<3}{marca}| {r['texto_itemset']:<{ancho_columna}} | {r['cobertura']:<10} | {r['soporte']:<10.6f} | {r['estado']:<15} |\n"

            pie = "-" * ancho_tabla + "\n"
            salida_fase1.append(encabezado_tabla + cuerpo + pie)

        return texto + "".join(salida_fase1)

    def ejecutar_fase2(self) -> str:
        """FASE 2: Extracción y Evaluación de Reglas de Asociación."""
        self.reglas_evaluadas = []

        # Generar reglas a partir de ítem-sets frecuentes de nivel k >= 2
        for nivel in sorted(self.itemsets_frecuentes_por_nivel.keys()):
            if nivel < 2:
                continue

            itemsets_nivel = self.itemsets_frecuentes_por_nivel[nivel]
            for conjunto_items, cobertura_conjunta in itemsets_nivel.items():
                elementos = list(conjunto_items)

                # Generar todas las particiones disjuntas en Antecedente (A) y Consecuente (B)
                for r in range(1, len(elementos)):
                    for tupla_antecedente in self._generar_combinaciones(elementos, r):
                        antecedente = frozenset(tupla_antecedente)
                        consecuente = conjunto_items - antecedente

                        cobertura_antecedente = self._calcular_cobertura_itemset(antecedente)
                        confianza = cobertura_conjunta / cobertura_antecedente if cobertura_antecedente > 0 else 0.0

                        aprobado_base = confianza >= self.confianza_minima_base
                        aprobado_estricto = confianza >= self.confianza_estricta

                        # Texto descriptivo de la regla
                        texto_antecedente = " AND ".join(sorted(list(antecedente)))
                        texto_consecuente = " AND ".join(sorted(list(consecuente)))
                        texto_regla = f"SI {texto_antecedente} ENTONCES {texto_consecuente}"

                        self.reglas_evaluadas.append({
                            'conjunto_origen': conjunto_items,
                            'antecedente': antecedente,
                            'consecuente': consecuente,
                            'texto_antecedente': texto_antecedente,
                            'texto_consecuente': texto_consecuente,
                            'texto_regla': texto_regla,
                            'cobertura_conjunta': cobertura_conjunta,
                            'cobertura_antecedente': cobertura_antecedente,
                            'confianza': confianza,
                            'soporte_regla': cobertura_conjunta / self.N,
                            'aprobado_base': aprobado_base,
                            'aprobado_estricto': aprobado_estricto,
                            'estado_base': "VÁLIDA" if aprobado_base else "DESCARTADA"
                        })

        # Ordenar reglas por confianza descendente y luego por cobertura conjunta descendente
        self.reglas_evaluadas = sorted(
            self.reglas_evaluadas,
            key=lambda r: (round(r['confianza'], 8), r['cobertura_conjunta']),
            reverse=True
        )

        reglas_validas_base = [r for r in self.reglas_evaluadas if r['aprobado_base']]
        reglas_descartadas_base = [r for r in self.reglas_evaluadas if not r['aprobado_base']]

        texto = "\n" + "=" * 90 + "\n"
        texto += "         FASE 2: EXTRACCIÓN Y EVALUACIÓN DE REGLAS DE ASOCIACIÓN\n"
        texto += "=" * 90 + "\n"
        texto += f" Total de Reglas Generadas y Evaluadas: {len(self.reglas_evaluadas)}\n"
        texto += f" Reglas Válidas (Confianza >= {self.confianza_minima_base:.2f}): {len(reglas_validas_base)}\n"
        texto += f" Reglas Descartadas:                    {len(reglas_descartadas_base)}\n"
        texto += "=" * 90 + "\n"

        if self.reglas_evaluadas:
            longitud_maxima = max(len(r['texto_regla']) for r in self.reglas_evaluadas)
            ancho_columna = max(55, longitud_maxima + 2)
            ancho_tabla = ancho_columna + 55

            encabezado_tabla = f"\n>>> TABLA DE EVALUACIÓN DE REGLAS DE ASOCIACIÓN (Confianza Mínima = {self.confianza_minima_base:.2f})\n"
            encabezado_tabla += "-" * ancho_tabla + "\n"
            encabezado_tabla += f"{'#':<3} | {'Regla de Asociación (A -> B)':<{ancho_columna}} | {'|A&B|':<6} | {'|A|':<6} | {'Confianza':<10} | {'Veredicto':<12} |\n"
            encabezado_tabla += "-" * ancho_tabla + "\n"

            cuerpo = ""
            for idx, r in enumerate(self.reglas_evaluadas, 1):
                marca = " [X]" if r['aprobado_base'] else "    "
                cuerpo += f"{idx:<3}{marca}| {r['texto_regla']:<{ancho_columna}} | {r['cobertura_conjunta']:<6} | {r['cobertura_antecedente']:<6} | {r['confianza']:<10.6f} | {r['estado_base']:<12} |\n"

            pie = "-" * ancho_tabla + "\n"
            texto += encabezado_tabla + cuerpo + pie

        return texto

    def ejecutar_resumen(self) -> str:
        """MEJORES REGLAS DESCUBIERTAS."""
        reglas_validas_base = [r for r in self.reglas_evaluadas if r['aprobado_base']]
        reglas_perfectas = [r for r in reglas_validas_base if round(r['confianza'], 6) == 1.0]
        reglas_altas = [r for r in reglas_validas_base if round(r['confianza'], 6) < 1.0]

        texto = "\n" + "=" * 90 + "\n"
        texto += "                 MEJORES REGLAS DESCUBIERTAS Y RESUMEN FINAL\n"
        texto += "=" * 90 + "\n"

        if reglas_perfectas:
            texto += f"\n REGLAS PERFECTAS CON CONFIANZA 1.0 ({len(reglas_perfectas)}):\n"
            for r in reglas_perfectas:
                texto += f"   - {r['texto_regla']}\n"
                texto += f"     -> Cobertura Conjunta (|A & B|): {r['cobertura_conjunta']} | Cobertura Antecedente (|A|): {r['cobertura_antecedente']} | Confianza: 1.000000\n"

        if reglas_altas:
            texto += f"\n REGLAS DE ALTA CONFIANZA ({self.confianza_minima_base:.2f} <= Confianza < 1.0) ({len(reglas_altas)}):\n"
            for r in reglas_altas:
                texto += f"   - {r['texto_regla']}\n"
                texto += f"     -> Cobertura Conjunta (|A & B|): {r['cobertura_conjunta']} | Cobertura Antecedente (|A|): {r['cobertura_antecedente']} | Confianza: {r['confianza']:.6f}\n"

        texto += "=" * 90 + "\n"
        return texto

    def ejecutar_todo(self) -> str:
        """Ejecuta el flujo completo del algoritmo APRIORI y genera el reporte general."""
        encabezado = self.generar_encabezado()
        print(encabezado)

        reporte_fase0 = self.ejecutar_fase0()
        print(reporte_fase0)

        reporte_fase1 = self.ejecutar_fase1()
        print(reporte_fase1)

        reporte_fase2 = self.ejecutar_fase2()
        print(reporte_fase2)

        reporte_resumen = self.ejecutar_resumen()
        print(reporte_resumen)

        return encabezado + reporte_fase0 + reporte_fase1 + reporte_fase2 + reporte_resumen


if __name__ == "__main__":
    archivo_csv = "dataset_apriori_supermercado(1).csv"
    clasificador = ClasificadorApriori(
        ruta_archivo=archivo_csv,
        soporte_minimo_pct=0.70,
        confianza_minima_base=0.85,
        confianza_estricta=0.90
    )
    clasificador.ejecutar_todo()
