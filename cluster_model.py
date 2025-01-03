import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

warnings.filterwarnings('ignore')

class PersonaModel:
    def __init__(self):
        self.clustering_features = [
            'Analytical thinking', 'Innovation', 'Leadership', 'Persistence',
            'Resilience', 'Rigor', 'Risk-taking', 'Self-control', 'Social orientation',
            'Stress tolerance', 'Integrity', 'Initiative', 'Independence'
        ]
        
        self.predictor_vars = [
            'Academic_Division', 'sector', 'Years since graduation', 'Job location',
            'Change in career', 'Degree of independence', 'Level of responsibility'
        ]
        
        self.categorical_features = [
            'Academic_Division', 'sector', 'Job location', 'Change in career'
        ]
        
        self.numerical_features = [col for col in self.predictor_vars 
                                   if col not in self.categorical_features]
        
        self.scaler = StandardScaler()
        self.kmeans = None
        self.classifier = None
        self.onehot_encoder = OneHotEncoder(handle_unknown='ignore')
        self.label_encoders = {}
        self.persona_descriptions = {}

    def preprocess_data(self, data):
        print("Starting data preprocessing...")
        processed_data = data.copy()
        
        print(f"Initial data shape: {processed_data.shape}")
        
        processed_data.replace([-99, '-99', '', ' '], np.nan, inplace=True)
        
        unnamed_cols = [col for col in processed_data.columns if 'Unnamed' in col]
        if unnamed_cols:
            processed_data.drop(columns=unnamed_cols, inplace=True)
            print(f"Dropped {len(unnamed_cols)} unnamed columns")
        
        processed_data = processed_data.loc[:, ~processed_data.columns.duplicated()]
        
        numeric_columns = [col for col in processed_data.columns 
                           if col in self.clustering_features + self.numerical_features]
        print(f"Numeric columns to process: {numeric_columns}")
        
        for col in numeric_columns:
            if col in processed_data.columns:
                processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
                print(f"Converted {col} to numeric. Null values: {processed_data[col].isnull().sum()}")
        
        numeric_imputer = SimpleImputer(strategy='median')
        numeric_cols = processed_data[numeric_columns].columns
        processed_data[numeric_cols] = numeric_imputer.fit_transform(processed_data[numeric_cols])
        print(f"Imputed {len(numeric_cols)} numeric columns")
        
        for col in self.categorical_features:
            if col in processed_data.columns:
                mode_values = processed_data[col].mode()
                if len(mode_values) > 0:
                    mode_value = mode_values.iloc[0]
                else:
                    mode_value = "Unknown"
                processed_data[col].fillna(mode_value, inplace=True)
                
                le = LabelEncoder()
                processed_data[col] = le.fit_transform(processed_data[col].astype(str))
                self.label_encoders[col] = le
                print(f"Processed categorical column: {col}")
        
        print(f"Final data shape: {processed_data.shape}")
        return processed_data

    def create_ensemble_model(self):
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        
        gb = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )
        
        ensemble = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb)],
            voting='soft',
            n_jobs=-1
        )
        return ensemble

    def analyze_cluster(self, cluster_data, cluster_id, available_clustering_features):
        mean_scores = cluster_data[available_clustering_features].mean()
        std_scores = cluster_data[available_clustering_features].std()
        
        top_traits = mean_scores.nlargest(5)
        bottom_traits = mean_scores.nsmallest(3)
        
        return {
            'size': len(cluster_data),
            'top_traits': dict(top_traits),
            'bottom_traits': dict(bottom_traits),
            'std_scores': dict(std_scores)
        }

    def generate_persona_description(self, cluster_profile):
        top_traits = list(cluster_profile['top_traits'].keys())
        trait_scores = cluster_profile['top_traits']
        
        if 'Innovation' in top_traits and 'Analytical thinking' in top_traits:
            base = "Research-Oriented Innovator"
            detail = "Strong in analytical thinking and innovation, suited for research and development roles"
        elif 'Leadership' in top_traits and 'Social orientation' in top_traits:
            base = "Leadership-Focused Professional"
            detail = "Strong leadership qualities with good social skills, suited for management and team leadership"
        elif 'Initiative' in top_traits and ('Risk-taking' in top_traits or 'Independence' in top_traits):
            base = "Industry Explorer"
            detail = "Shows strong initiative and independence, well-suited for entrepreneurial roles"
        else:
            base = f"Professional"
            detail = f"Characterized by strong {', '.join(top_traits[:2]).lower()}"
        
        return f"{base}: {detail}"

    def fit(self, data):
        try:
            print("\nStarting model fitting...")
            
            processed_data = self.preprocess_data(data)
            
            available_clustering_features = [f for f in self.clustering_features 
                                             if f in processed_data.columns]
            print(f"\nAvailable clustering features: {available_clustering_features}")
            
            if not available_clustering_features:
                raise ValueError("No clustering features available in the data")
            
            clustering_data = processed_data[available_clustering_features]
            clustering_scaled = self.scaler.fit_transform(clustering_data)
            print(f"Clustering data shape: {clustering_scaled.shape}")
            
            min_clusters = 3
            max_clusters = min(10, len(clustering_scaled))
            silhouette_scores = []
            k_range = range(min_clusters, max_clusters + 1)
            
            print("\nEvaluating cluster numbers...")
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(clustering_scaled)
                score = silhouette_score(clustering_scaled, labels)
                print(f"k={k}, silhouette score={score:.3f}")
                silhouette_scores.append((k, score))
            
            best_k = max(silhouette_scores, key=lambda x: x[1])[0]
            optimal_k = best_k
            print(f"\nSelected optimal number of clusters: {optimal_k}")
            
            self.kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            processed_data['Persona'] = self.kmeans.fit_predict(clustering_scaled)
            processed_data['Persona'] = processed_data['Persona'].astype('category')
            
            print("\nAnalyzing cluster characteristics...")
            for i in range(optimal_k):
                cluster_mask = processed_data['Persona'] == i
                cluster_data = processed_data[cluster_mask]
                
                cluster_profile = self.analyze_cluster(
                    cluster_data, i, available_clustering_features
                )
                
                self.persona_descriptions[i] = self.generate_persona_description(cluster_profile)
                
                print(f"\nPersona {i} ({cluster_profile['size']} members):")
                print(f"Description: {self.persona_descriptions[i]}")
                print("Top traits:")
                for trait, value in cluster_profile['top_traits'].items():
                    print(f"  {trait}: {value:.2f}")
            
            self.available_cat_features = [f for f in self.categorical_features 
                                           if f in processed_data.columns]
            self.available_num_features = [f for f in self.numerical_features 
                                           if f in processed_data.columns]
            
            # Modified feature matrix preparation
            X_num = processed_data[self.available_num_features].values
            X_cat = self.onehot_encoder.fit_transform(processed_data[self.available_cat_features]).toarray()
            
            print(f"X_num shape: {X_num.shape}")
            print(f"X_cat shape: {X_cat.shape}")
            
            if X_num.size == 0 and X_cat.size == 0:
                raise ValueError("No predictor variables available for training the classifier.")
            elif X_num.size == 0:
                X = X_cat
            elif X_cat.size == 0:
                X = X_num
            else:
                X = np.hstack([X_num, X_cat])
            y = processed_data['Persona']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            self.classifier = self.create_ensemble_model()
            self.classifier.fit(X_train, y_train)
            
            y_pred = self.classifier.predict(X_test)
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
            print("Confusion Matrix:")
            conf_matrix = confusion_matrix(y_test, y_pred)
            print(conf_matrix)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                        xticklabels=self.classifier.classes_, yticklabels=self.classifier.classes_)
            plt.xlabel('Predicted Persona')
            plt.ylabel('True Persona')
            plt.title('Confusion Matrix')
            plt.show()
            
            print("\nModel fitting completed successfully")
            return self
            
        except Exception as e:
            print(f"Error during model fitting: {str(e)}")
            import traceback
            print("Traceback:")
            print(traceback.format_exc())
            raise

    def predict(self, data):
        try:
            if not all([self.kmeans, self.classifier, self.scaler]):
                raise ValueError("Model not fitted. Call fit() first.")

            processed_data = self.preprocess_data(data)

            # Modified prediction feature matrix preparation
            X_num = processed_data[self.available_num_features].values
            X_cat = self.onehot_encoder.transform(processed_data[self.available_cat_features]).toarray()

            print(f"X_num shape: {X_num.shape}")
            print(f"X_cat shape: {X_cat.shape}")

            if X_num.size == 0 and X_cat.size == 0:
                raise ValueError("No predictor variables available for prediction.")
            elif X_num.size == 0:
                X = X_cat
            elif X_cat.size == 0:
                X = X_num
            else:
                X = np.hstack([X_num, X_cat])

            predictions = self.classifier.predict(X)
            probabilities = self.classifier.predict_proba(X)
            confidence_scores = np.max(probabilities, axis=1)

            predicted_personas = predictions.astype(int)
            persona_descriptions = [self.persona_descriptions.get(p, "") for p in predicted_personas]

            results = pd.DataFrame({
                'Predicted_Persona': predictions,
                'Confidence_Score': confidence_scores,
                'Persona_Description': persona_descriptions
            })

            return results

        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            import traceback
            print("Traceback:")
            print(traceback.format_exc())
            raise

    def save_model(self, path_prefix='models/'):
        try:
            joblib.dump(self.kmeans, f'{path_prefix}kmeans_model.joblib')
            joblib.dump(self.classifier, f'{path_prefix}classifier_model.joblib')
            joblib.dump(self.scaler, f'{path_prefix}scaler.joblib')
            joblib.dump(self.onehot_encoder, f'{path_prefix}onehot_encoder.joblib')
            joblib.dump(self.label_encoders, f'{path_prefix}label_encoders.joblib')
            joblib.dump({
                'num_features': self.available_num_features,
                'cat_features': self.available_cat_features,
                'persona_descriptions': self.persona_descriptions
            }, f'{path_prefix}model_metadata.joblib')
            print("Model saved successfully")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            raise

    @classmethod
    def load_model(cls, path_prefix='models/'):
        try:
            instance = cls()
            instance.kmeans = joblib.load(f'{path_prefix}kmeans_model.joblib')
            instance.classifier = joblib.load(f'{path_prefix}classifier_model.joblib')
            instance.scaler = joblib.load(f'{path_prefix}scaler.joblib')
            instance.onehot_encoder = joblib.load(f'{path_prefix}onehot_encoder.joblib')
            instance.label_encoders = joblib.load(f'{path_prefix}label_encoders.joblib')
            
            metadata = joblib.load(f'{path_prefix}model_metadata.joblib')
            instance.available_num_features = metadata['num_features']
            instance.available_cat_features = metadata['cat_features']
            instance.persona_descriptions = metadata['persona_descriptions']
            
            print("Model loaded successfully")
            return instance
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        print("Loading data...")
        alumni_data = pd.read_csv('alumni_data.csv')
        print(f"Loaded data shape: {alumni_data.shape}")
        
        print("\nInitializing model...")
        model = PersonaModel()
        
        print("\nFitting model...")
        model.fit(alumni_data)
        
        print("\nMaking predictions on sample data...")
        test_data = alumni_data.head()
        results = model.predict(test_data)
        
        print("\nPrediction Results:")
        for idx, row in results.iterrows():
            print(f"\nSample {idx + 1}:")
            print(f"  Persona: {row['Persona_Description']}")
            print(f"  Confidence Score: {row['Confidence_Score']:.2f}")
        
        # Save the model
        print("\nSaving model...")
        model.save_model()
            
    except Exception as e:
        print(f"An error occurred during execution: {str(e)}")
        import traceback
        print("\nFull traceback:")
        print(traceback.format_exc())
