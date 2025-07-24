import requests
import json
import time
from typing import List, Dict, Any

# Configuration
API_BASE_URL = "http://127.0.0.1:8000"  # Sesuaikan dengan URL API Anda
OUTPUT_DIR = "collected_data"  # Folder untuk menyimpan hasil
TOP_K = 1  # Jumlah rekomendasi per query

# Data lists
skill_list = [
    "Accounting", "Android", "Artificial Intelligence", "Cloud Computing",
    "Communication Skills", "Cybersecurity", "Excel", "Graphic Design",
    "HR", "iOS", "Java", "JavaScript", "Machine Learning", "Marketing",
    "Math", "Network", "Networking", "OOP", "Power BI", "Presentation Skills",
    "Project Management", "Python", "SEO", "Social Media", "SQL",
    "Statistics", "Web Development"
]

popular_majors = [
    "Teknik Informatika",
    "Sistem Informasi", 
    "Teknik Komputer",
    "Ilmu Komputer",
    "Manajemen",
    "Akuntansi",
    "Ekonomi",
    "Matematika",
    "Psikologi",
    "Komunikasi",
    "Desain Grafis"
]

job_titles = [
    "Data Analyst",
    "Backend Developer", 
    "QA Tester",
    "Social Media Strategist",
    "Public Relations",
    "Tax Consultant",
    "Copywriter",
    "Financial Analyst",
    "Auditor",
    "HR Generalist",
    "Career Counselor",
    "User Researcher"
]

class CourseRecommendationCollector:
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def test_api_connection(self) -> bool:
        """Test if API is running and accessible"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                health_data = response.json()
                print(f"API Connection Success: {health_data['status']}")
                return True
            else:
                print(f"API Health Check Failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"API Connection Error: {e}")
            return False
    
    def get_skill_recommendations(self, skill: str, top_k: int = TOP_K) -> Dict[str, Any]:
        """Get course recommendations for a skill query"""
        try:
            payload = {
                "query": f"I want to learn {skill}",
                "top_k": top_k
            }
            
            response = self.session.post(f"{self.base_url}/recommend", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                print(f"Skill '{skill}': {data['total_results']} recommendations")
                return {
                    "skill": skill,
                    "status": "success",
                    "data": data
                }
            else:
                print(f"Skill '{skill}' failed: {response.status_code}")
                return {
                    "skill": skill,
                    "status": "error",
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            print(f"Skill '{skill}' exception: {e}")
            return {
                "skill": skill,
                "status": "error",
                "error": str(e)
            }
    
    def get_job_recommendations(self, job_title: str, top_k: int = TOP_K) -> Dict[str, Any]:
        """Get course recommendations for a job title"""
        try:
            payload = {
                "job_title": job_title,
                "top_k": top_k
            }
            
            response = self.session.post(f"{self.base_url}/recommend-by-job", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                print(f"Job '{job_title}': {data['total_results']} recommendations")
                return {
                    "job_title": job_title,
                    "status": "success",
                    "data": data
                }
            else:
                print(f"Job '{job_title}' failed: {response.status_code}")
                return {
                    "job_title": job_title,
                    "status": "error",
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            print(f"Job '{job_title}' exception: {e}")
            return {
                "job_title": job_title,
                "status": "error",
                "error": str(e)
            }
    
    def get_major_recommendations(self, major: str, top_k: int = TOP_K) -> Dict[str, Any]:
        """Get course recommendations for a major/field of study"""
        try:
            payload = {
                "query": f"I am studying {major} and want to improve my skills",
                "top_k": top_k
            }
            
            response = self.session.post(f"{self.base_url}/recommend", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                print(f"Major '{major}': {data['total_results']} recommendations")
                return {
                    "major": major,
                    "status": "success",
                    "data": data
                }
            else:
                print(f"Major '{major}' failed: {response.status_code}")
                return {
                    "major": major,
                    "status": "error",
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            print(f"Major '{major}' exception: {e}")
            return {
                "major": major,
                "status": "error",
                "error": str(e)
            }
    
    def collect_all_data(self, delay: float = 0.5) -> Dict[str, List[Dict]]:
        """Collect all recommendations with delay between requests"""
        print(" Starting data collection...")
        
        # Test API connection first
        if not self.test_api_connection():
            return {"error": "API connection failed"}
        
        results = {
            "skills": [],
            "jobs": [],
            "majors": [],
            "summary": {}
        }
        
        # Collect skill-based recommendations
        print(f"\nCollecting skill recommendations ({len(skill_list)} skills)...")
        for skill in skill_list:
            result = self.get_skill_recommendations(skill)
            results["skills"].append(result)
            time.sleep(delay)  # Rate limiting
        
        # Collect job-based recommendations
        print(f"\nCollecting job recommendations ({len(job_titles)} jobs)...")
        for job in job_titles:
            result = self.get_job_recommendations(job)
            results["jobs"].append(result)
            time.sleep(delay)  # Rate limiting
        
        # Collect major-based recommendations
        print(f"\n Collecting major recommendations ({len(popular_majors)} majors)...")
        for major in popular_majors:
            result = self.get_major_recommendations(major)
            results["majors"].append(result)
            time.sleep(delay)  # Rate limiting
        
        # Generate summary
        results["summary"] = self._generate_summary(results)
        
        print("\n Data collection completed!")
        return results
    
    def _generate_summary(self, results: Dict) -> Dict[str, Any]:
        """Generate summary statistics"""
        summary = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "categories": {}
        }
        
        for category, data_list in results.items():
            if category == "summary":
                continue
                
            total = len(data_list)
            successful = len([item for item in data_list if item.get("status") == "success"])
            failed = total - successful
            
            summary["categories"][category] = {
                "total": total,
                "successful": successful,
                "failed": failed,
                "success_rate": f"{(successful/total*100):.1f}%" if total > 0 else "0%"
            }
            
            summary["total_queries"] += total
            summary["successful_queries"] += successful
            summary["failed_queries"] += failed
        
        summary["overall_success_rate"] = f"{(summary['successful_queries']/summary['total_queries']*100):.1f}%" if summary["total_queries"] > 0 else "0%"
        
        return summary
    
    def save_results(self, results: Dict, filename: str = "course_recommendations_data.json"):
        """Save results to JSON file"""
        try:
            import os
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            
            filepath = os.path.join(OUTPUT_DIR, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f" Data saved to: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"❌ Error saving data: {e}")
            return None
    
    def save_separate_files(self, results: Dict):
        """Save each category to separate JSON files"""
        import os
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        files_created = []
        
        for category, data in results.items():
            if category == "summary":
                continue
                
            filename = f"{category}_recommendations.json"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                files_created.append(filepath)
                print(f" {category.title()} data saved to: {filepath}")
            except Exception as e:
                print(f"❌ Error saving {category} data: {e}")
        
        # Save summary separately
        try:
            summary_path = os.path.join(OUTPUT_DIR, "collection_summary.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(results["summary"], f, indent=2, ensure_ascii=False)
            files_created.append(summary_path)
            print(f" Summary saved to: {summary_path}")
        except Exception as e:
            print(f"❌ Error saving summary: {e}")
        
        return files_created

def main():
    """Main execution function"""
    print(" Course Recommendation Data Collector")
    print("=" * 50)
    
    # Initialize collector
    collector = CourseRecommendationCollector()
    
    # Collect all data
    all_results = collector.collect_all_data(delay=0.3)  # 300ms delay between requests
    
    if "error" in all_results:
        print(f"❌ Collection failed: {all_results['error']}")
        return
    
    # Print summary
    print("\n Collection Summary:")
    print("-" * 30)
    summary = all_results["summary"]
    print(f"Total Queries: {summary['total_queries']}")
    print(f"Successful: {summary['successful_queries']}")
    print(f"Failed: {summary['failed_queries']}")
    print(f"Success Rate: {summary['overall_success_rate']}")
    
    print("\nBy Category:")
    for category, stats in summary["categories"].items():
        print(f"  {category.title()}: {stats['successful']}/{stats['total']} ({stats['success_rate']})")
    
    # Save results
    print(f"\n Saving results...")
    
    # Save complete data
    main_file = collector.save_results(all_results)
    
    # Save separate files
    separate_files = collector.save_separate_files(all_results)
    
    print(f"\n Collection complete! Files created:")
    if main_file:
        print(f"  Complete data: {main_file}")
    for file in separate_files:
        print(f"  {file}")

if __name__ == "__main__":
    main()