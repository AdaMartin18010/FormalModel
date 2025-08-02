#!/bin/bash

# 形式化模型平台实施启动脚本
# Formal Model Platform Implementation Startup Script

echo "🚀 启动形式化模型平台实施..."
echo "Starting Formal Model Platform Implementation..."

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 检查必要的工具
check_requirements() {
    echo -e "${BLUE}📋 检查系统要求...${NC}"
    
    # 检查 Git
    if ! command -v git &> /dev/null; then
        echo -e "${RED}❌ Git 未安装，请先安装 Git${NC}"
        exit 1
    fi
    
    # 检查 Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker 未安装，请先安装 Docker${NC}"
        exit 1
    fi
    
    # 检查 Node.js
    if ! command -v node &> /dev/null; then
        echo -e "${RED}❌ Node.js 未安装，请先安装 Node.js${NC}"
        exit 1
    fi
    
    # 检查 Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python3 未安装，请先安装 Python3${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ 所有要求已满足${NC}"
}

# 创建项目结构
create_project_structure() {
    echo -e "${BLUE}📁 创建项目结构...${NC}"
    
    # 创建主目录
    mkdir -p formal-model-platform
    cd formal-model-platform
    
    # 创建子目录
    mkdir -p {frontend,backend,ai,devops,docs,scripts,tests}
    mkdir -p frontend/{src,public,components,pages,hooks,utils,types}
    mkdir -p backend/{src,config,database,middleware,controllers,services,validators}
    mkdir -p ai/{src,models,services,utils}
    mkdir -p devops/{k8s,docker,scripts,monitoring}
    mkdir -p docs/{api,user-guide,developer-guide}
    
    echo -e "${GREEN}✅ 项目结构创建完成${NC}"
}

# 初始化 Git 仓库
init_git_repository() {
    echo -e "${BLUE}🔧 初始化 Git 仓库...${NC}"
    
    git init
    git remote add origin https://github.com/formal-model/platform.git
    
    # 创建 .gitignore
    cat > .gitignore << EOF
# Dependencies
node_modules/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.env

# Build outputs
dist/
build/
*.tsbuildinfo

# Logs
*.log
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Runtime data
pids
*.pid
*.seed
*.pid.lock

# Coverage directory used by tools like istanbul
coverage/
*.lcov

# nyc test coverage
.nyc_output

# Dependency directories
jspm_packages/

# Optional npm cache directory
.npm

# Optional eslint cache
.eslintcache

# Microbundle cache
.rpt2_cache/
.rts2_cache_cjs/
.rts2_cache_es/
.rts2_cache_umd/

# Optional REPL history
.node_repl_history

# Output of 'npm pack'
*.tgz

# Yarn Integrity file
.yarn-integrity

# dotenv environment variables file
.env
.env.test
.env.production

# parcel-bundler cache (https://parceljs.org/)
.cache
.parcel-cache

# next.js build output
.next

# nuxt.js build output
.nuxt

# vuepress build output
.vuepress/dist

# Serverless directories
.serverless/

# FuseBox cache
.fusebox/

# DynamoDB Local files
.dynamodb/

# TernJS port file
.tern-port

# Stores VSCode versions used for testing VSCode extensions
.vscode-test

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Docker
.dockerignore

# Kubernetes
*.kubeconfig

# Terraform
*.tfstate
*.tfstate.*
.terraform/

# Local development
local/
EOF
    
    git add .
    git commit -m "Initial commit: Project structure setup"
    
    echo -e "${GREEN}✅ Git 仓库初始化完成${NC}"
}

# 创建 Docker 配置
create_docker_config() {
    echo -e "${BLUE}🐳 创建 Docker 配置...${NC}"
    
    # 创建 docker-compose.yml
    cat > docker-compose.yml << EOF
version: '3.8'

services:
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=development
      - REACT_APP_API_URL=http://localhost:8000
    volumes:
      - ./frontend:/app
      - /app/node_modules
    depends_on:
      - backend

  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - NODE_ENV=development
      - DATABASE_URL=postgresql://user:pass@db:5432/formal_model
      - REDIS_URL=redis://redis:6379
      - JWT_SECRET=your-secret-key
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
    volumes:
      - ./backend:/app
      - /app/node_modules
    depends_on:
      - db
      - redis

  ai-service:
    build: ./ai
    ports:
      - "8001:8001"
    environment:
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
      - CLAUDE_API_KEY=\${CLAUDE_API_KEY}
    volumes:
      - ./ai:/app
    depends_on:
      - backend

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=formal_model
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./devops/nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./devops/nginx/ssl:/etc/nginx/ssl
    depends_on:
      - frontend
      - backend

volumes:
  postgres_data:
  redis_data:
EOF

    # 创建前端 Dockerfile
    cat > frontend/Dockerfile << EOF
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .

EXPOSE 3000

CMD ["npm", "start"]
EOF

    # 创建后端 Dockerfile
    cat > backend/Dockerfile << EOF
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .

EXPOSE 8000

CMD ["npm", "start"]
EOF

    # 创建 AI 服务 Dockerfile
    cat > ai/Dockerfile << EOF
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8001

CMD ["python", "app.py"]
EOF

    echo -e "${GREEN}✅ Docker 配置创建完成${NC}"
}

# 创建前端配置
create_frontend_config() {
    echo -e "${BLUE}⚛️ 创建前端配置...${NC}"
    
    cd frontend
    
    # 创建 package.json
    cat > package.json << EOF
{
  "name": "formal-model-frontend",
  "version": "1.0.0",
  "description": "Formal Model Platform Frontend",
  "private": true,
  "dependencies": {
    "@tanstack/react-query": "^5.0.0",
    "@types/node": "^20.0.0",
    "@types/react": "^18.0.0",
    "@types/react-dom": "^18.0.0",
    "axios": "^1.6.0",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.8.0",
    "react-scripts": "5.0.1",
    "typescript": "^5.0.0",
    "tailwindcss": "^3.3.0",
    "autoprefixer": "^10.4.0",
    "postcss": "^8.4.0",
    "lucide-react": "^0.294.0",
    "clsx": "^2.0.0",
    "tailwind-merge": "^2.0.0"
  },
  "devDependencies": {
    "@testing-library/jest-dom": "^6.1.0",
    "@testing-library/react": "^14.0.0",
    "@testing-library/user-event": "^14.5.0",
    "@types/jest": "^29.5.0"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject",
    "lint": "eslint src --ext .ts,.tsx",
    "lint:fix": "eslint src --ext .ts,.tsx --fix"
  },
  "eslintConfig": {
    "extends": [
      "react-app",
      "react-app/jest"
    ]
  },
  "browserslist": {
    "production": [
      ">0.2%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 1 chrome version",
      "last 1 firefox version",
      "last 1 safari version"
    ]
  }
}
EOF

    # 创建 TypeScript 配置
    cat > tsconfig.json << EOF
{
  "compilerOptions": {
    "target": "es5",
    "lib": [
      "dom",
      "dom.iterable",
      "es6"
    ],
    "allowJs": true,
    "skipLibCheck": true,
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "strict": true,
    "forceConsistentCasingInFileNames": true,
    "noFallthroughCasesInSwitch": true,
    "module": "esnext",
    "moduleResolution": "node",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx"
  },
  "include": [
    "src"
  ]
}
EOF

    # 创建 Tailwind CSS 配置
    cat > tailwind.config.js << EOF
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#eff6ff',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
        }
      }
    },
  },
  plugins: [],
}
EOF

    # 创建 PostCSS 配置
    cat > postcss.config.js << EOF
module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
EOF

    cd ..
    echo -e "${GREEN}✅ 前端配置创建完成${NC}"
}

# 创建后端配置
create_backend_config() {
    echo -e "${BLUE}🔧 创建后端配置...${NC}"
    
    cd backend
    
    # 创建 package.json
    cat > package.json << EOF
{
  "name": "formal-model-backend",
  "version": "1.0.0",
  "description": "Formal Model Platform Backend",
  "main": "dist/index.js",
  "scripts": {
    "start": "node dist/index.js",
    "dev": "nodemon src/index.ts",
    "build": "tsc",
    "test": "jest",
    "test:watch": "jest --watch",
    "lint": "eslint src --ext .ts",
    "lint:fix": "eslint src --ext .ts --fix"
  },
  "dependencies": {
    "express": "^4.18.0",
    "cors": "^2.8.5",
    "helmet": "^7.0.0",
    "morgan": "^1.10.0",
    "dotenv": "^16.3.0",
    "bcryptjs": "^2.4.3",
    "jsonwebtoken": "^9.0.0",
    "joi": "^17.9.0",
    "pg": "^8.11.0",
    "redis": "^4.6.0",
    "axios": "^1.6.0",
    "openai": "^4.0.0",
    "anthropic": "^0.7.0",
    "multer": "^1.4.5",
    "compression": "^1.7.4"
  },
  "devDependencies": {
    "@types/express": "^4.17.0",
    "@types/cors": "^2.8.0",
    "@types/morgan": "^1.9.0",
    "@types/bcryptjs": "^2.4.0",
    "@types/jsonwebtoken": "^9.0.0",
    "@types/multer": "^1.4.0",
    "@types/compression": "^1.7.0",
    "@types/node": "^20.0.0",
    "@types/pg": "^8.10.0",
    "@types/jest": "^29.5.0",
    "typescript": "^5.0.0",
    "ts-node": "^10.9.0",
    "nodemon": "^3.0.0",
    "jest": "^29.5.0",
    "ts-jest": "^29.1.0",
    "eslint": "^8.0.0",
    "@typescript-eslint/eslint-plugin": "^6.0.0",
    "@typescript-eslint/parser": "^6.0.0"
  }
}
EOF

    # 创建 TypeScript 配置
    cat > tsconfig.json << EOF
{
  "compilerOptions": {
    "target": "es2020",
    "module": "commonjs",
    "lib": ["es2020"],
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist", "**/*.test.ts"]
}
EOF

    # 创建 Jest 配置
    cat > jest.config.js << EOF
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  roots: ['<rootDir>/src'],
  testMatch: ['**/__tests__/**/*.ts', '**/?(*.)+(spec|test).ts'],
  transform: {
    '^.+\\.ts$': 'ts-jest',
  },
  collectCoverageFrom: [
    'src/**/*.ts',
    '!src/**/*.d.ts',
  ],
};
EOF

    cd ..
    echo -e "${GREEN}✅ 后端配置创建完成${NC}"
}

# 创建 AI 服务配置
create_ai_config() {
    echo -e "${BLUE}🤖 创建 AI 服务配置...${NC}"
    
    cd ai
    
    # 创建 requirements.txt
    cat > requirements.txt << EOF
fastapi==0.104.0
uvicorn==0.24.0
pydantic==2.5.0
python-dotenv==1.0.0
openai==1.3.0
anthropic==0.7.0
redis==5.0.0
psycopg2-binary==2.9.0
sqlalchemy==2.0.0
alembic==1.13.0
pytest==7.4.0
pytest-asyncio==0.21.0
httpx==0.25.0
python-multipart==0.0.6
python-jose==3.3.0
passlib==1.7.4
bcrypt==4.1.0
EOF

    # 创建主应用文件
    cat > app.py << EOF
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Formal Model AI Service",
    description="AI service for formal model analysis and generation",
    version="1.0.0"
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 安全配置
security = HTTPBearer()

class ModelAnalysisRequest(BaseModel):
    description: str
    category: Optional[str] = None

class ModelAnalysisResponse(BaseModel):
    model_type: str
    parameters: List[str]
    mathematical_formula: str
    applications: List[str]
    implementation_suggestions: List[str]

@app.get("/")
async def root():
    return {"message": "Formal Model AI Service"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/analyze", response_model=ModelAnalysisResponse)
async def analyze_model(request: ModelAnalysisRequest):
    try:
        # TODO: 实现 AI 分析逻辑
        return ModelAnalysisResponse(
            model_type="Differential Equation",
            parameters=["time", "position", "velocity"],
            mathematical_formula="dx/dt = v",
            applications=["Physics", "Engineering"],
            implementation_suggestions=["Use numerical methods", "Consider boundary conditions"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
EOF

    cd ..
    echo -e "${GREEN}✅ AI 服务配置创建完成${NC}"
}

# 创建环境变量文件
create_env_files() {
    echo -e "${BLUE}🔐 创建环境变量文件...${NC}"
    
    # 创建 .env.example
    cat > .env.example << EOF
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/formal_model
REDIS_URL=redis://localhost:6379

# JWT
JWT_SECRET=your-secret-key-here
JWT_EXPIRES_IN=7d

# AI Services
OPENAI_API_KEY=your-openai-api-key
CLAUDE_API_KEY=your-claude-api-key

# Server
PORT=8000
NODE_ENV=development

# Frontend
REACT_APP_API_URL=http://localhost:8000
REACT_APP_AI_SERVICE_URL=http://localhost:8001
EOF

    # 创建 .env (开发环境)
    cp .env.example .env
    
    echo -e "${GREEN}✅ 环境变量文件创建完成${NC}"
    echo -e "${YELLOW}⚠️  请编辑 .env 文件，填入正确的 API 密钥${NC}"
}

# 创建 CI/CD 配置
create_cicd_config() {
    echo -e "${BLUE}🔄 创建 CI/CD 配置...${NC}"
    
    mkdir -p .github/workflows
    
    # 创建 GitHub Actions 配置
    cat > .github/workflows/ci.yml << EOF
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test-backend:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      redis:
        image: redis:7-alpine
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379

    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
        cache: 'npm'
        cache-dependency-path: backend/package-lock.json
    
    - name: Install backend dependencies
      run: |
        cd backend
        npm ci
    
    - name: Run backend tests
      run: |
        cd backend
        npm test
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db
        REDIS_URL: redis://localhost:6379

  test-frontend:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
        cache: 'npm'
        cache-dependency-path: frontend/package-lock.json
    
    - name: Install frontend dependencies
      run: |
        cd frontend
        npm ci
    
    - name: Run frontend tests
      run: |
        cd frontend
        npm test -- --watchAll=false
    
    - name: Build frontend
      run: |
        cd frontend
        npm run build

  test-ai:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install AI dependencies
      run: |
        cd ai
        pip install -r requirements.txt
    
    - name: Run AI tests
      run: |
        cd ai
        pytest

  deploy:
    needs: [test-backend, test-frontend, test-ai]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to production
      run: |
        echo "Deploying to production..."
        # TODO: 添加部署脚本
EOF

    echo -e "${GREEN}✅ CI/CD 配置创建完成${NC}"
}

# 创建 README 文件
create_readme() {
    echo -e "${BLUE}📖 创建 README 文件...${NC}"
    
    cat > README.md << EOF
# 形式化模型平台 / Formal Model Platform

## 项目概述 / Project Overview

形式化模型平台是一个全面的、标准化的形式化模型知识体系，旨在为学术界、工业界和教育界提供高质量的理论基础和实用工具。

## 技术栈 / Tech Stack

### 前端 / Frontend
- React 18 + TypeScript
- Tailwind CSS
- React Query
- React Router

### 后端 / Backend
- Node.js + Express
- TypeScript
- PostgreSQL
- Redis
- JWT 认证

### AI 服务 / AI Service
- Python + FastAPI
- OpenAI API
- Claude API
- Redis 缓存

### 基础设施 / Infrastructure
- Docker + Docker Compose
- Kubernetes
- Nginx
- Prometheus + Grafana

## 快速开始 / Quick Start

### 环境要求 / Requirements
- Node.js 18+
- Python 3.11+
- Docker
- Git

### 安装步骤 / Installation

1. 克隆仓库
\`\`\`bash
git clone https://github.com/formal-model/platform.git
cd platform
\`\`\`

2. 配置环境变量
\`\`\`bash
cp .env.example .env
# 编辑 .env 文件，填入正确的 API 密钥
\`\`\`

3. 启动开发环境
\`\`\`bash
docker-compose up -d
\`\`\`

4. 安装依赖
\`\`\`bash
# 前端依赖
cd frontend && npm install

# 后端依赖
cd ../backend && npm install

# AI 服务依赖
cd ../ai && pip install -r requirements.txt
\`\`\`

5. 启动服务
\`\`\`bash
# 前端 (http://localhost:3000)
cd frontend && npm start

# 后端 (http://localhost:8000)
cd backend && npm run dev

# AI 服务 (http://localhost:8001)
cd ai && python app.py
\`\`\`

## 开发指南 / Development Guide

### 项目结构 / Project Structure
\`\`\`
formal-model-platform/
├── frontend/          # React 前端应用
├── backend/           # Node.js 后端 API
├── ai/               # Python AI 服务
├── devops/           # 运维配置
├── docs/             # 文档
└── scripts/          # 脚本文件
\`\`\`

### 开发流程 / Development Workflow

1. 创建功能分支
\`\`\`bash
git checkout -b feature/your-feature-name
\`\`\`

2. 开发功能
\`\`\`bash
# 前端开发
cd frontend && npm start

# 后端开发
cd backend && npm run dev
\`\`\`

3. 运行测试
\`\`\`bash
# 前端测试
cd frontend && npm test

# 后端测试
cd backend && npm test

# AI 服务测试
cd ai && pytest
\`\`\`

4. 提交代码
\`\`\`bash
git add .
git commit -m "feat: add your feature"
git push origin feature/your-feature-name
\`\`\`

## API 文档 / API Documentation

- 后端 API: http://localhost:8000/docs
- AI 服务 API: http://localhost:8001/docs

## 贡献指南 / Contributing

请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解如何贡献代码。

## 许可证 / License

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 联系方式 / Contact

- 项目主页: https://github.com/formal-model/platform
- 问题反馈: https://github.com/formal-model/platform/issues
- 讨论区: https://github.com/formal-model/platform/discussions
EOF

    echo -e "${GREEN}✅ README 文件创建完成${NC}"
}

# 主函数
main() {
    echo -e "${GREEN}🎉 形式化模型平台实施启动脚本${NC}"
    echo "=========================================="
    
    # 检查系统要求
    check_requirements
    
    # 创建项目结构
    create_project_structure
    
    # 初始化 Git 仓库
    init_git_repository
    
    # 创建 Docker 配置
    create_docker_config
    
    # 创建前端配置
    create_frontend_config
    
    # 创建后端配置
    create_backend_config
    
    # 创建 AI 服务配置
    create_ai_config
    
    # 创建环境变量文件
    create_env_files
    
    # 创建 CI/CD 配置
    create_cicd_config
    
    # 创建 README 文件
    create_readme
    
    echo ""
    echo -e "${GREEN}🎉 项目初始化完成！${NC}"
    echo ""
    echo -e "${BLUE}📋 下一步操作：${NC}"
    echo "1. 编辑 .env 文件，填入正确的 API 密钥"
    echo "2. 运行 'docker-compose up -d' 启动服务"
    echo "3. 访问 http://localhost:3000 查看前端"
    echo "4. 访问 http://localhost:8000/docs 查看后端 API"
    echo "5. 访问 http://localhost:8001/docs 查看 AI 服务 API"
    echo ""
    echo -e "${YELLOW}💡 提示：请确保已安装 Docker 并启动 Docker 服务${NC}"
}

# 运行主函数
main "$@" 