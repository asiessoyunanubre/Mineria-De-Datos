import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuración para reproducibilidad
np.random.seed(42)
random.seed(42)

class StudentDropoutDataset:
    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
    def generate_demographic_data(self, n_records):
        """Genera datos demográficos con variables categóricas y numéricas"""
        print("Generando datos demográficos...")
        
        # Variables categóricas
        genders = ['Male', 'Female', 'Non-binary']
        ethnicities = ['White', 'Hispanic', 'Black', 'Asian', 'Mixed', 'Other']
        locations = ['Urban', 'Suburban', 'Rural']
        parental_education = ['High School', 'Some College', 'Bachelor', 'Master', 'PhD', 'No Education']
        
        data = {
            'student_id': [f'STU{str(i).zfill(4)}' for i in range(1, n_records + 1)],
            'age': np.random.normal(20, 2, n_records).astype(int),
            'gender': np.random.choice(genders, n_records, p=[0.45, 0.45, 0.1]),
            'ethnicity': np.random.choice(ethnicities, n_records, p=[0.5, 0.2, 0.15, 0.1, 0.03, 0.02]),
            'location': np.random.choice(locations, n_records, p=[0.6, 0.3, 0.1]),
            'parental_education': np.random.choice(parental_education, n_records, p=[0.3, 0.25, 0.2, 0.15, 0.08, 0.02]),
            'distance_from_campus': np.random.exponential(10, n_records),
            'first_generation': np.random.choice([0, 1], n_records, p=[0.7, 0.3])
        }
        
        # Agregar valores nulos (5% en algunas variables)
        null_indices = np.random.choice(n_records, size=int(n_records * 0.05), replace=False)
        for idx in null_indices:
            data['parental_education'][idx] = None
            
        null_indices_gender = np.random.choice(n_records, size=int(n_records * 0.02), replace=False)
        for idx in null_indices_gender:
            data['gender'][idx] = None
            
        return data
    
    def generate_academic_data(self, n_records):
        """Genera datos académicos con outliers y valores atípicos"""
        print("Generando datos académicos...")
        
        # Variables categóricas académicas
        majors = ['Computer Science', 'Business', 'Engineering', 'Psychology', 
                 'Biology', 'Arts', 'Mathematics', 'Economics', 'Political Science']
        enrollment_status = ['Full-time', 'Part-time']
        academic_level = ['Freshman', 'Sophomore', 'Junior', 'Senior']
        
        # Generar GPA de escuela secundaria
        high_school_gpa = np.clip(np.random.normal(3.2, 0.4, n_records), 1.0, 4.0)
        
        # Generar GPA actual con más variabilidad
        current_gpa = []
        for i in range(n_records):
            base_gpa = np.random.normal(2.8, 0.6)
            current_gpa.append(max(0.0, min(4.0, base_gpa)))
        
        data = {
            'high_school_gpa': high_school_gpa,
            'current_gpa': current_gpa,
            'major': np.random.choice(majors, n_records, p=[0.2, 0.15, 0.15, 0.1, 0.1, 0.08, 0.08, 0.07, 0.07]),
            'credits_completed': np.random.poisson(45, n_records),
            'credits_current': np.random.choice([12, 15, 18, 9, 6], n_records, p=[0.4, 0.3, 0.1, 0.15, 0.05]),
            'enrollment_status': np.random.choice(enrollment_status, n_records, p=[0.8, 0.2]),
            'academic_level': np.random.choice(academic_level, n_records, p=[0.25, 0.25, 0.25, 0.25]),
            'failed_courses': np.random.poisson(1, n_records),
            'library_visits_per_week': np.random.poisson(2, n_records),
            'study_hours_per_week': np.clip(np.random.normal(15, 8, n_records), 0, 40)
        }
        
        # Agregar outliers en GPA (3% de los registros)
        outlier_indices = np.random.choice(n_records, size=int(n_records * 0.03), replace=False)
        for idx in outlier_indices:
            data['current_gpa'][idx] = random.choice([-1.0, 5.0, 10.0])
            
        # Agregar valores nulos (3% en current_gpa)
        null_indices = np.random.choice(n_records, size=int(n_records * 0.03), replace=False)
        for idx in null_indices:
            data['current_gpa'][idx] = None
            
        # Agregar outliers en créditos completados (2%)
        outlier_credits = np.random.choice(n_records, size=int(n_records * 0.02), replace=False)
        for idx in outlier_credits:
            data['credits_completed'][idx] = random.choice([-50, 200, 500])
            
        return data
    
    def generate_financial_data(self, n_records):
        """Genera datos financieros con outliers realistas"""
        print("Generando datos financieros...")
        
        # Variables categóricas financieras
        financial_aid_types = ['Scholarship', 'Loan', 'Work-Study', 'Grant', 'None', 'Multiple']
        employment_status = ['Unemployed', 'Part-time', 'Full-time']
        
        # Distribución de ingresos familiares (con outliers)
        family_income = np.random.lognormal(10.5, 0.8, n_records)
        
        data = {
            'family_income': family_income,
            'tuition_owed': np.random.exponential(2000, n_records),
            'financial_aid_amount': np.random.exponential(5000, n_records),
            'financial_aid_type': np.random.choice(financial_aid_types, n_records, p=[0.25, 0.25, 0.1, 0.15, 0.2, 0.05]),
            'employment_status': np.random.choice(employment_status, n_records, p=[0.4, 0.4, 0.2]),
            'work_hours_per_week': np.random.poisson(10, n_records),
            'scholarship_amount': np.random.exponential(2000, n_records),
            'out_of_pocket_expenses': np.random.exponential(1000, n_records)
        }
        
        # Agregar outliers extremos en ingresos familiares (4%)
        outlier_income = np.random.choice(n_records, size=int(n_records * 0.04), replace=False)
        for idx in outlier_income:
            data['family_income'][idx] = random.choice([1000000, 5000000, 10000000])
            
        # Agregar outliers bajos en ingresos (3%)
        low_income_outliers = np.random.choice(n_records, size=int(n_records * 0.03), replace=False)
        for idx in low_income_outliers:
            data['family_income'][idx] = random.uniform(100, 1000)
            
        # Agregar valores nulos en ayuda financiera (4%)
        null_financial = np.random.choice(n_records, size=int(n_records * 0.04), replace=False)
        for idx in null_financial:
            data['financial_aid_type'][idx] = None
            
        return data
    
    def calculate_dropout_probability(self, df):
        """Calcula la probabilidad de deserción basada en múltiples factores"""
        print("Calculando probabilidades de deserción...")
        
        probabilities = []
        
        for idx in range(len(df)):
            prob = 0.0
            
            # Usar .iloc para acceso seguro por posición
            row = df.iloc[idx]
            
            # Factores académicos (40% de peso)
            current_gpa = row['current_gpa']
            if pd.notna(current_gpa):
                if current_gpa < 2.0:
                    prob += 0.25
                elif current_gpa < 2.5:
                    prob += 0.15
                elif current_gpa > 3.5:
                    prob -= 0.1
                    
            if row['failed_courses'] > 3:
                prob += 0.15
            if row['study_hours_per_week'] < 5:
                prob += 0.1
                
            # Factores financieros (35% de peso)
            if row['family_income'] < 20000:
                prob += 0.2
            if row['tuition_owed'] > 5000:
                prob += 0.15
            if row['financial_aid_type'] in ['None', 'Loan']:
                prob += 0.1
            if row['employment_status'] == 'Full-time':
                prob += 0.05
                
            # Factores demográficos (25% de peso)
            if row['first_generation'] == 1:
                prob += 0.1
            if row['distance_from_campus'] > 30:
                prob += 0.08
            if row['parental_education'] in ['High School', 'No Education']:
                prob += 0.07
                
            # Ruido aleatorio
            prob += np.random.normal(0, 0.1)
            
            probabilities.append(max(0, min(1, prob)))
            
        return probabilities
    
    def generate_dataset(self, n_records=500):
        """Genera el dataset completo"""
        print(f"Iniciando generación de dataset con {n_records} registros...")
        
        # Generar todos los datos
        demographic_data = self.generate_demographic_data(n_records)
        academic_data = self.generate_academic_data(n_records)
        financial_data = self.generate_financial_data(n_records)
        
        # Combinar todos los datos
        df = pd.DataFrame({**demographic_data, **academic_data, **financial_data})
        
        # Calcular probabilidades de deserción
        dropout_probs = self.calculate_dropout_probability(df)
        
        # Generar variable objetivo basada en probabilidades
        dropout = []
        for prob in dropout_probs:
            # Ajustar umbral para obtener distribución realista (~30% deserción)
            threshold = 0.3 + np.random.normal(0, 0.05)
            if prob > threshold:
                dropout.append(1)
            else:
                dropout.append(0)
                
        df['dropout'] = dropout
        
        print(f"Dataset generado exitosamente!")
        print(f"Tasa de deserción: {df['dropout'].mean():.2%}")
        
        return df

def create_and_save_dataset():
    """Función simplificada para crear y guardar el dataset"""
    
    print("=== GENERADOR DE DATASET DE DESERCIÓN ESTUDIANTIL ===")
    
    # Crear generador
    generator = StudentDropoutDataset(seed=42)
    
    # Generar dataset
    df = generator.generate_dataset(n_records=500)
    
    # Mostrar información básica
    print("\n=== INFORMACIÓN DEL DATASET ===")
    print(f"Total de registros: {len(df)}")
    print(f"Total de variables: {len(df.columns)}")
    print(f"Tasa de deserción: {df['dropout'].mean():.2%}")
    
    # Mostrar valores nulos
    print("\n=== VALORES NULOS ===")
    null_counts = df.isnull().sum()
    for col, count in null_counts.items():
        if count > 0:
            print(f"{col}: {count} valores nulos ({count/len(df):.1%})")
    
    # Mostrar tipos de datos
    print("\n=== TIPOS DE DATOS ===")
    print(df.dtypes)
    
    # Guardar dataset
    filename = 'student_dropout_dataset.csv'
    df.to_csv(filename, index=False)
    print(f"\n✅ Dataset guardado como: {filename}")
    
    # Mostrar primeras filas
    print("\n=== PRIMERAS 5 FILAS DEL DATASET ===")
    print(df.head())
    
    return df

def quick_analysis(df):
    """Análisis rápido del dataset"""
    print("\n=== ANÁLISIS RÁPIDO ===")
    
    # Estadísticas básicas
    print("\nEstadísticas descriptivas:")
    print(df.describe())
    
    # Distribución de variables categóricas clave
    categorical_vars = ['gender', 'ethnicity', 'major', 'academic_level', 'financial_aid_type']
    
    for var in categorical_vars:
        if var in df.columns:
            print(f"\nDistribución de {var}:")
            print(df[var].value_counts())
    
    # Detección de outliers en variables numéricas clave
    numeric_vars = ['current_gpa', 'family_income', 'credits_completed']
    
    print("\n=== DETECCIÓN DE OUTLIERS ===")
    for var in numeric_vars:
        if var in df.columns:
            Q1 = df[var].quantile(0.25)
            Q3 = df[var].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[var] < lower_bound) | (df[var] > upper_bound)]
            print(f"{var}: {len(outliers)} outliers ({len(outliers)/len(df):.1%})")

# Ejecutar el código
if __name__ == "__main__":
    try:
        df = create_and_save_dataset()
        quick_analysis(df)
        print("\n🎉 ¡Dataset creado exitosamente!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Posible solución: Asegúrate de tener pandas, numpy y matplotlib instalados")
        print("   Puedes instalarlos con: pip install pandas numpy matplotlib seaborn")