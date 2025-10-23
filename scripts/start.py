#!/usr/bin/env python3
"""
Forest Cover Type Prediction - Main Startup Script
This script provides a unified entry point for the forest cover prediction system.
It can start the FastAPI server, run training, or perform other operations.
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

# Add the project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("Checking dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'fastapi', 'uvicorn',
        'lightgbm', 'xgboost', 'torch', 'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"[OK] {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"[MISSING] {package}")
    
    if missing_packages:
        print(f"\n[WARNING] Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    print("[SUCCESS] All dependencies are installed!")
    return True

def check_model_files():
    """Check if model files exist"""
    print("\nChecking model files...")
    
    models_dir = PROJECT_ROOT / "models"
    required_models = [
        "best_model_lightgbm.pkl",
        "quick_optimized_model.pkl"
    ]
    
    missing_models = []
    
    for model_file in required_models:
        model_path = models_dir / model_file
        if model_path.exists():
            print(f"[OK] {model_file}")
        else:
            missing_models.append(model_file)
            print(f"[MISSING] {model_file}")
    
    if missing_models:
        print(f"\n[WARNING] Missing model files: {', '.join(missing_models)}")
        print("Run training first with: python start.py --train")
        return False
    
    print("[SUCCESS] All model files are present!")
    return True

def start_server(host="0.0.0.0", port=8001, reload=False, frontend_port=3000):
    """Start the FastAPI server with optional frontend development server"""
    print(f"\n[STARTING] FastAPI server on {host}:{port}")
    print("[INFO] Frontend will be available at: http://localhost:8001")
    print("[INFO] API documentation at: http://localhost:8001/docs")
    print("[INFO] Health check at: http://localhost:8001/health")
    
    # Check if we should start a separate frontend dev server
    frontend_dev_server = False
    try:
        # Check if we have a package.json for frontend development
        package_json = PROJECT_ROOT / "frontend" / "package.json"
        if package_json.exists():
            print(f"[INFO] Frontend development server available on port {frontend_port}")
            frontend_dev_server = True
    except:
        pass
    
    print("\nPress Ctrl+C to stop the server")
    
    try:
        import uvicorn
        from fastapi_main import app
        
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n[INFO] Server stopped by user")
    except Exception as e:
        print(f"\n[ERROR] Error starting server: {e}")
        sys.exit(1)

def start_fullstack(backend_host="0.0.0.0", backend_port=8001, frontend_port=3000, reload=False):
    """Start both frontend and backend servers"""
    print(f"\n[STARTING] Full-stack application...")
    print(f"[INFO] Backend API: http://localhost:{backend_port}")
    print(f"[INFO] Frontend: http://localhost:{frontend_port}")
    print(f"[INFO] API docs: http://localhost:{backend_port}/docs")
    print("\nPress Ctrl+C to stop all servers")
    
    import subprocess
    import threading
    import time
    
    # Start backend server in a separate thread
    def start_backend():
        try:
            import uvicorn
            from fastapi_main import app
            uvicorn.run(
                app,
                host=backend_host,
                port=backend_port,
                reload=reload,
                log_level="info"
            )
        except Exception as e:
            print(f"[ERROR] Backend server error: {e}")
    
    # Start frontend development server if available
    def start_frontend():
        try:
            # Try to start a simple HTTP server for the frontend
            import http.server
            import socketserver
            import os
            
            os.chdir(PROJECT_ROOT / "frontend")
            handler = http.server.SimpleHTTPRequestHandler
            with socketserver.TCPServer(("", frontend_port), handler) as httpd:
                print(f"[INFO] Frontend server started on port {frontend_port}")
                httpd.serve_forever()
        except Exception as e:
            print(f"[ERROR] Frontend server error: {e}")
    
    try:
        # Start backend in a separate thread
        backend_thread = threading.Thread(target=start_backend, daemon=True)
        backend_thread.start()
        
        # Wait a moment for backend to start
        time.sleep(2)
        
        # Start frontend in main thread
        start_frontend()
        
    except KeyboardInterrupt:
        print("\n[INFO] Stopping all servers...")
    except Exception as e:
        print(f"[ERROR] Error starting full-stack: {e}")

def run_training():
    """Run model training"""
    print("\n[STARTING] Model training...")
    
    try:
        # Check if training script exists
        train_script = PROJECT_ROOT / "train_models.py"
        if not train_script.exists():
            print("[ERROR] Training script not found: train_models.py")
            return False
        
        # Run training
        result = subprocess.run([sys.executable, str(train_script)], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("[SUCCESS] Training completed successfully!")
            print(result.stdout)
        else:
            print("[ERROR] Training failed!")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"[ERROR] Error running training: {e}")
        return False
    
    return True

def run_optimization():
    """Run model optimization"""
    print("\n[STARTING] Model optimization...")
    
    try:
        # Check if optimization script exists
        opt_script = PROJECT_ROOT / "quick_optimization.py"
        if not opt_script.exists():
            print("[ERROR] Optimization script not found: quick_optimization.py")
            return False
        
        # Run optimization
        result = subprocess.run([sys.executable, str(opt_script)], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("[SUCCESS] Optimization completed successfully!")
            print(result.stdout)
        else:
            print("[ERROR] Optimization failed!")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"[ERROR] Error running optimization: {e}")
        return False
    
    return True

def run_analysis():
    """Run data analysis"""
    print("\n[STARTING] Data analysis...")
    
    try:
        # Check if analysis script exists
        analysis_script = PROJECT_ROOT / "quick_analysis.py"
        if not analysis_script.exists():
            print("[ERROR] Analysis script not found: quick_analysis.py")
            return False
        
        # Run analysis
        result = subprocess.run([sys.executable, str(analysis_script)], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("[SUCCESS] Analysis completed successfully!")
            print(result.stdout)
        else:
            print("[ERROR] Analysis failed!")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"[ERROR] Error running analysis: {e}")
        return False
    
    return True

def show_status():
    """Show system status"""
    print("\nSystem Status:")
    print("=" * 50)
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Check model files
    models_ok = check_model_files()
    
    # Check data files
    data_file = PROJECT_ROOT / "train.csv"
    print(f"\nData file: {'[OK]' if data_file.exists() else '[MISSING]'} train.csv")
    
    # Check frontend
    frontend_dir = PROJECT_ROOT / "frontend"
    print(f"Frontend: {'[OK]' if frontend_dir.exists() else '[MISSING]'} frontend/")
    
    # Overall status
    print("\n" + "=" * 50)
    if deps_ok and models_ok and data_file.exists():
        print("[SUCCESS] System is ready to run!")
    else:
        print("[WARNING] System needs setup. Run with --setup flag for help.")
    
    return deps_ok and models_ok and data_file.exists()

def setup_system():
    """Setup the system"""
    print("\n[SETUP] Setting up Forest Cover Prediction System...")
    print("=" * 60)
    
    # Check if virtual environment exists
    venv_path = PROJECT_ROOT / "venv"
    if not venv_path.exists():
        print("[ERROR] Virtual environment not found!")
        print("Please create a virtual environment first:")
        print("  python -m venv venv")
        print("  venv\\Scripts\\activate  # Windows")
        print("  source venv/bin/activate  # Linux/Mac")
        return False
    
    # Install dependencies
    print("\n[INSTALLING] Dependencies...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("[SUCCESS] Dependencies installed successfully!")
        else:
            print("[ERROR] Failed to install dependencies!")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"[ERROR] Error installing dependencies: {e}")
        return False
    
    # Run training if models don't exist
    if not check_model_files():
        print("\n[STARTING] Training models...")
        if not run_training():
            print("[ERROR] Training failed!")
            return False
    
    print("\n[SUCCESS] Setup completed successfully!")
    return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Forest Cover Type Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start.py                    # Start the server
  python start.py --server           # Start the server explicitly
  python start.py --fullstack        # Start both frontend and backend
  python start.py --train            # Train models
  python start.py --optimize         # Optimize models
  python start.py --analyze          # Run data analysis
  python start.py --status           # Show system status
  python start.py --setup             # Setup the system
  python start.py --server --reload   # Start server with auto-reload
  python start.py --fullstack --frontend-port 3001  # Custom frontend port
        """
    )
    
    parser.add_argument("--server", action="store_true", 
                       help="Start the FastAPI server")
    parser.add_argument("--fullstack", action="store_true", 
                       help="Start both frontend and backend servers")
    parser.add_argument("--train", action="store_true", 
                       help="Train the models")
    parser.add_argument("--optimize", action="store_true", 
                       help="Optimize the models")
    parser.add_argument("--analyze", action="store_true", 
                       help="Run data analysis")
    parser.add_argument("--status", action="store_true", 
                       help="Show system status")
    parser.add_argument("--setup", action="store_true", 
                       help="Setup the system")
    parser.add_argument("--host", default="0.0.0.0", 
                       help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8001, 
                       help="Server port (default: 8001)")
    parser.add_argument("--reload", action="store_true", 
                       help="Enable auto-reload for development")
    parser.add_argument("--frontend-port", type=int, default=3000, 
                       help="Frontend server port (default: 3000)")
    
    args = parser.parse_args()
    
    # Print banner
    print("Forest Cover Type Prediction System")
    print("=" * 50)
    
    # Handle different commands
    if args.setup:
        setup_system()
    elif args.status:
        show_status()
    elif args.train:
        run_training()
    elif args.optimize:
        run_optimization()
    elif args.analyze:
        run_analysis()
    elif args.fullstack:
        # Check if system is ready
        if not show_status():
            print("\n[WARNING] System not ready. Run with --setup to fix issues.")
            sys.exit(1)
        
        # Start full-stack application
        start_fullstack(
            backend_host=args.host, 
            backend_port=args.port, 
            frontend_port=args.frontend_port,
            reload=args.reload
        )
    elif args.server or len(sys.argv) == 1:  # Default to server
        # Check if system is ready
        if not show_status():
            print("\n⚠️  System not ready. Run with --setup to fix issues.")
            sys.exit(1)
        
        # Start server
        start_server(host=args.host, port=args.port, reload=args.reload)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
