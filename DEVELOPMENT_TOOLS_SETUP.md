# 开发工具和环境设置指南 / Development Tools and Environment Setup Guide

## 项目概述 / Project Overview

**项目名称**: 形式化模型平台开发工具设置 / Formal Model Platform Development Tools Setup  
**设置目标**: 建立完整的开发环境和工具链  
**适用范围**: 开发团队、测试团队、运维团队  
**更新时间**: 2025-08-01  

## 开发环境设置 / Development Environment Setup

### 1. 本地开发环境 / Local Development Environment

#### 1.1 基础工具安装 / Basic Tools Installation

**操作系统要求**:

- Windows 10/11 或 macOS 10.15+ 或 Ubuntu 20.04+
- 内存: 16GB+ (推荐32GB)
- 存储: 100GB+ 可用空间
- 网络: 稳定的互联网连接

**必需工具**:

```bash
# 1. Git 版本控制
# Windows: 下载 Git for Windows
# macOS: brew install git
# Ubuntu: sudo apt-get install git

# 2. Node.js 环境
# 推荐版本: 18.x LTS
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 18
nvm use 18

# 3. Python 环境
# 推荐版本: 3.11+
# Windows: 下载 Python 3.11
# macOS: brew install python@3.11
# Ubuntu: sudo apt-get install python3.11

# 4. Docker 容器化
# Windows: 下载 Docker Desktop
# macOS: brew install docker
# Ubuntu: sudo apt-get install docker.io

# 5. IDE 推荐
# VS Code: https://code.visualstudio.com/
# IntelliJ IDEA: https://www.jetbrains.com/idea/
# PyCharm: https://www.jetbrains.com/pycharm/
```

#### 1.2 项目环境配置 / Project Environment Configuration

**前端环境设置**:

```bash
# 克隆项目
git clone https://github.com/formal-model/platform.git
cd platform/frontend

# 安装依赖
npm install

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，配置必要的环境变量

# 启动开发服务器
npm run dev
```

**后端环境设置**:

```bash
# 进入后端目录
cd ../backend

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件

# 启动开发服务器
python app.py
```

**数据库环境设置**:

```bash
# 安装 PostgreSQL
# Windows: 下载 PostgreSQL 安装包
# macOS: brew install postgresql
# Ubuntu: sudo apt-get install postgresql postgresql-contrib

# 启动 PostgreSQL 服务
# Windows: 通过服务管理器启动
# macOS: brew services start postgresql
# Ubuntu: sudo systemctl start postgresql

# 创建数据库
createdb formal_model_dev

# 运行数据库迁移
cd backend
python manage.py migrate
```

#### 1.3 开发工具配置 / Development Tools Configuration

**VS Code 配置**:

```json
// .vscode/settings.json
{
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.fixAll": true,
    "source.organizeImports": true
  },
  "typescript.preferences.importModuleSpecifier": "relative",
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black",
  "files.exclude": {
    "**/node_modules": true,
    "**/__pycache__": true,
    "**/.pytest_cache": true
  },
  "extensions.recommendations": [
    "ms-vscode.vscode-typescript-next",
    "ms-python.python",
    "ms-python.black-formatter",
    "ms-python.pylint",
    "bradlc.vscode-tailwindcss",
    "esbenp.prettier-vscode"
  ]
}
```

**Git 配置**:

```bash
# 配置 Git 用户信息
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# 配置 Git 别名
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.st status

# 配置 Git 钩子
# 在项目根目录创建 .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

### 2. CI/CD 流水线设置 / CI/CD Pipeline Setup

#### 2.1 GitHub Actions 配置 / GitHub Actions Configuration

```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
        cache: 'npm'
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        cd frontend && npm ci
        cd ../backend && pip install -r requirements.txt
    
    - name: Run tests
      run: |
        cd frontend && npm test
        cd ../backend && python -m pytest
    
    - name: Run linting
      run: |
        cd frontend && npm run lint
        cd ../backend && black --check .
        cd ../backend && pylint app/
    
    - name: Build frontend
      run: cd frontend && npm run build
    
    - name: Build Docker images
      run: |
        docker build -t formal-model/frontend:${{ github.sha }} ./frontend
        docker build -t formal-model/backend:${{ github.sha }} ./backend

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to staging
      run: |
        # 部署到测试环境
        echo "Deploying to staging..."
    
    - name: Run integration tests
      run: |
        # 运行集成测试
        echo "Running integration tests..."
    
    - name: Deploy to production
      run: |
        # 部署到生产环境
        echo "Deploying to production..."
```

#### 2.2 Docker 配置 / Docker Configuration

**前端 Dockerfile**:

```dockerfile
# frontend/Dockerfile
FROM node:18-alpine as builder

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

**后端 Dockerfile**:

```dockerfile
# backend/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建非root用户
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
```

**Docker Compose 配置**:

```yaml
# docker-compose.yml
version: '3.8'

services:
  frontend:
    build: ./frontend
    ports:
      - "3000:80"
    depends_on:
      - backend
    environment:
      - REACT_APP_API_URL=http://localhost:8000

  backend:
    build: ./backend
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - redis
    environment:
      - DATABASE_URL=postgresql://user:password@postgres:5432/formal_model
      - REDIS_URL=redis://redis:6379
      - SECRET_KEY=your-secret-key

  postgres:
    image: postgres:13
    environment:
      - POSTGRES_DB=formal_model
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

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
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - frontend
      - backend

volumes:
  postgres_data:
  redis_data:
```

### 3. 监控和日志系统 / Monitoring and Logging System

#### 3.1 Prometheus 监控 / Prometheus Monitoring

**Prometheus 配置**:

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'formal-model-backend'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'formal-model-frontend'
    static_configs:
      - targets: ['frontend:80']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

**Grafana 仪表板**:

```json
// grafana-dashboard.json
{
  "dashboard": {
    "title": "Formal Model Platform",
    "panels": [
      {
        "title": "API Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m])",
            "legendFormat": "5xx errors"
          }
        ]
      },
      {
        "title": "Active Users",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(active_users_total)",
            "legendFormat": "Active Users"
          }
        ]
      }
    ]
  }
}
```

#### 3.2 ELK 日志系统 / ELK Logging System

**Elasticsearch 配置**:

```yaml
# elasticsearch.yml
cluster.name: formal-model-cluster
node.name: node-1
network.host: 0.0.0.0
http.port: 9200
discovery.seed_hosts: ["elasticsearch"]
cluster.initial_master_nodes: ["node-1"]
```

**Logstash 配置**:

```ruby
# logstash.conf
input {
  beats {
    port => 5044
  }
}

filter {
  if [fields][service] == "backend" {
    grok {
      match => { "message" => "%{TIMESTAMP_ISO8601:timestamp} %{LOGLEVEL:level} %{GREEDYDATA:message}" }
    }
    date {
      match => [ "timestamp", "ISO8601" ]
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "formal-model-%{+YYYY.MM.dd}"
  }
}
```

**Kibana 配置**:

```yaml
# kibana.yml
server.name: kibana
server.host: "0.0.0.0"
elasticsearch.hosts: ["http://elasticsearch:9200"]
monitoring.ui.container.elasticsearch.enabled: true
```

### 4. 测试工具配置 / Testing Tools Configuration

#### 4.1 单元测试 / Unit Testing

**前端测试配置**:

```javascript
// frontend/jest.config.js
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'jsdom',
  setupFilesAfterEnv: ['<rootDir>/src/setupTests.ts'],
  moduleNameMapping: {
    '^@/(.*)$': '<rootDir>/src/$1',
  },
  collectCoverageFrom: [
    'src/**/*.{ts,tsx}',
    '!src/**/*.d.ts',
    '!src/index.tsx',
  ],
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80,
    },
  },
};
```

**后端测试配置**:

```python
# backend/pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --verbose
    --tb=short
    --strict-markers
    --disable-warnings
    --cov=app
    --cov-report=html
    --cov-report=term-missing
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
```

#### 4.2 集成测试 / Integration Testing

**API 测试配置**:

```python
# backend/tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

class TestAPI:
    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}
    
    def test_create_model(self):
        model_data = {
            "name": "Test Model",
            "description": "A test model",
            "type": "mathematical"
        }
        response = client.post("/api/models", json=model_data)
        assert response.status_code == 201
        assert response.json()["name"] == "Test Model"
    
    def test_get_model(self):
        # 先创建模型
        model_data = {"name": "Test Model", "description": "Test"}
        create_response = client.post("/api/models", json=model_data)
        model_id = create_response.json()["id"]
        
        # 获取模型
        response = client.get(f"/api/models/{model_id}")
        assert response.status_code == 200
        assert response.json()["name"] == "Test Model"
```

#### 4.3 端到端测试 / End-to-End Testing

**Playwright 配置**:

```javascript
// frontend/playwright.config.js
module.exports = {
  testDir: './tests/e2e',
  timeout: 30000,
  expect: {
    timeout: 5000,
  },
  use: {
    baseURL: 'http://localhost:3000',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: { browserName: 'chromium' },
    },
    {
      name: 'firefox',
      use: { browserName: 'firefox' },
    },
    {
      name: 'webkit',
      use: { browserName: 'webkit' },
    },
  ],
  webServer: {
    command: 'npm run dev',
    port: 3000,
    reuseExistingServer: !process.env.CI,
  },
};
```

**E2E 测试示例**:

```typescript
// frontend/tests/e2e/model-creation.spec.ts
import { test, expect } from '@playwright/test';

test.describe('Model Creation', () => {
  test('should create a new model', async ({ page }) => {
    await page.goto('/');
    
    // 点击创建模型按钮
    await page.click('[data-testid="create-model-btn"]');
    
    // 填写模型信息
    await page.fill('[data-testid="model-name"]', 'Test Model');
    await page.fill('[data-testid="model-description"]', 'A test model');
    await page.selectOption('[data-testid="model-type"]', 'mathematical');
    
    // 提交表单
    await page.click('[data-testid="submit-btn"]');
    
    // 验证模型创建成功
    await expect(page.locator('[data-testid="success-message"]')).toBeVisible();
    await expect(page.locator('[data-testid="model-list"]')).toContainText('Test Model');
  });
  
  test('should validate required fields', async ({ page }) => {
    await page.goto('/');
    await page.click('[data-testid="create-model-btn"]');
    await page.click('[data-testid="submit-btn"]');
    
    // 验证错误信息
    await expect(page.locator('[data-testid="name-error"]')).toBeVisible();
    await expect(page.locator('[data-testid="description-error"]')).toBeVisible();
  });
});
```

### 5. 代码质量工具 / Code Quality Tools

#### 5.1 代码检查工具 / Code Linting Tools

**ESLint 配置**:

```javascript
// frontend/.eslintrc.js
module.exports = {
  extends: [
    'eslint:recommended',
    '@typescript-eslint/recommended',
    'react/recommended',
    'react-hooks/recommended',
  ],
  parser: '@typescript-eslint/parser',
  plugins: ['@typescript-eslint', 'react', 'react-hooks'],
  rules: {
    'react/react-in-jsx-scope': 'off',
    'react/prop-types': 'off',
    '@typescript-eslint/no-unused-vars': 'error',
    '@typescript-eslint/explicit-function-return-type': 'off',
    '@typescript-eslint/explicit-module-boundary-types': 'off',
  },
  settings: {
    react: {
      version: 'detect',
    },
  },
};
```

**Prettier 配置**:

```json
// frontend/.prettierrc
{
  "semi": true,
  "trailingComma": "es5",
  "singleQuote": true,
  "printWidth": 80,
  "tabWidth": 2,
  "useTabs": false
}
```

**Pylint 配置**:

```ini
# backend/.pylintrc
[MASTER]
disable=
    C0114, # missing-module-docstring
    C0115, # missing-class-docstring
    C0116, # missing-function-docstring

[FORMAT]
max-line-length=88

[MESSAGES CONTROL]
disable=C0111,R0903,C0103

[REPORTS]
output-format=text
score=yes
```

#### 5.2 代码覆盖率 / Code Coverage

**前端覆盖率配置**:

```javascript
// frontend/coverage.config.js
module.exports = {
  coverageDirectory: 'coverage',
  collectCoverageFrom: [
    'src/**/*.{js,jsx,ts,tsx}',
    '!src/**/*.d.ts',
    '!src/index.tsx',
    '!src/serviceWorker.ts',
  ],
  coverageReporters: ['text', 'lcov', 'html'],
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80,
    },
  },
};
```

**后端覆盖率配置**:

```ini
# backend/.coveragerc
[run]
source = app
omit = 
    */tests/*
    */migrations/*
    */__init__.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    if self.debug:
    if settings.DEBUG
    raise AssertionError
    raise NotImplementedError
    if 0:
    if __name__ == .__main__.:
    class .*\bProtocol\):
    @(abc\.)?abstractmethod
```

### 6. 安全工具配置 / Security Tools Configuration

#### 6.1 安全扫描 / Security Scanning

**OWASP ZAP 配置**:

```yaml
# security-scan.yml
name: Security Scan
on: [push, pull_request]

jobs:
  security-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Run OWASP ZAP Scan
      uses: zaproxy/action-full-scan@v0.7.0
      with:
        target: 'http://localhost:3000'
        rules_file_name: '.zap/rules.tsv'
        cmd_options: '-a'
```

**Snyk 配置**:

```yaml
# snyk-scan.yml
name: Snyk Security Scan
on: [push, pull_request]

jobs:
  snyk:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Run Snyk to check for vulnerabilities
      uses: snyk/actions/node@master
      env:
        SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
      with:
        args: --severity-threshold=high
```

#### 6.2 依赖检查 / Dependency Checking

**npm audit 配置**:

```json
// frontend/package.json
{
  "scripts": {
    "audit": "npm audit --audit-level=high",
    "audit:fix": "npm audit fix"
  },
  "husky": {
    "hooks": {
      "pre-commit": "npm run lint && npm run test",
      "pre-push": "npm run audit"
    }
  }
}
```

**Python 安全检查**:

```bash
# backend/security-check.sh
#!/bin/bash

# 检查 Python 依赖安全漏洞
safety check

# 检查已知漏洞
bandit -r app/

# 检查依赖许可证
pip-licenses --format=markdown > LICENSES.md
```

### 7. 性能测试工具 / Performance Testing Tools

#### 7.1 负载测试 / Load Testing

**Artillery 配置**:

```yaml
# performance/load-test.yml
config:
  target: 'http://localhost:8000'
  phases:
    - duration: 60
      arrivalRate: 10
      name: "Warm up"
    - duration: 300
      arrivalRate: 50
      name: "Sustained load"
    - duration: 60
      arrivalRate: 100
      name: "Peak load"
  defaults:
    headers:
      Content-Type: 'application/json'

scenarios:
  - name: "API endpoints"
    weight: 70
    flow:
      - get:
          url: "/api/health"
      - post:
          url: "/api/models"
          json:
            name: "Test Model"
            description: "Performance test model"
      - get:
          url: "/api/models"

  - name: "Model operations"
    weight: 30
    flow:
      - post:
          url: "/api/models"
          json:
            name: "{{ $randomString() }}"
            description: "{{ $randomString() }}"
      - think: 2
      - get:
          url: "/api/models"
```

**JMeter 配置**:

```xml
<!-- performance/jmeter-test.jmx -->
<?xml version="1.0" encoding="UTF-8"?>
<jmeterTestPlan version="1.2" properties="5.0">
  <hashTree>
    <TestPlan guiclass="TestPlanGui" testclass="TestPlan" testname="Formal Model API Test">
      <elementProp name="TestPlan.arguments" elementType="Arguments">
        <collectionProp name="Arguments.arguments"/>
      </elementProp>
      <boolProp name="TestPlan.functional_mode">false</boolProp>
      <stringProp name="TestPlan.comments"></stringProp>
      <boolProp name="TestPlan.tearDown_on_shutdown">true</boolProp>
      <boolProp name="TestPlan.serialize_threadgroups">false</boolProp>
    </TestPlan>
    <hashTree>
      <ThreadGroup guiclass="ThreadGroupGui" testclass="ThreadGroup" testname="API Load Test">
        <elementProp name="ThreadGroup.main_controller" elementType="LoopController">
          <boolProp name="LoopController.continue_forever">false</boolProp>
          <stringProp name="LoopController.loops">10</stringProp>
        </elementProp>
        <stringProp name="ThreadGroup.on_sample_error">continue</stringProp>
        <elementProp name="ThreadGroup.scheduler" elementType="ThreadGroupScheduler">
          <stringProp name="ThreadGroupScheduler.duration">300</stringProp>
          <stringProp name="ThreadGroupScheduler.delay">0</stringProp>
        </elementProp>
        <stringProp name="ThreadGroup.num_threads">50</stringProp>
        <stringProp name="ThreadGroup.ramp_time">10</stringProp>
        <boolProp name="ThreadGroup.scheduler">true</boolProp>
        <stringProp name="ThreadGroup.duration"></stringProp>
        <stringProp name="ThreadGroup.delay"></stringProp>
      </ThreadGroup>
      <hashTree>
        <HTTPSamplerProxy guiclass="HttpTestSampleGui" testclass="HTTPSamplerProxy">
          <elementProp name="HTTPsampler.Arguments">
            <collectionProp name="Arguments.arguments"/>
          </elementProp>
          <stringProp name="HTTPSampler.domain">localhost</stringProp>
          <stringProp name="HTTPSampler.port">8000</stringProp>
          <stringProp name="HTTPSampler.protocol">http</stringProp>
          <stringProp name="HTTPSampler.path">/api/health</stringProp>
          <stringProp name="HTTPSampler.method">GET</stringProp>
        </HTTPSamplerProxy>
        <hashTree/>
      </hashTree>
    </hashTree>
  </hashTree>
</jmeterTestPlan>
```

#### 7.2 性能监控 / Performance Monitoring

**Lighthouse CI 配置**:

```yaml
# .lighthouserc.js
module.exports = {
  ci: {
    collect: {
      url: ['http://localhost:3000'],
      numberOfRuns: 3,
    },
    assert: {
      assertions: {
        'categories:performance': ['warn', { minScore: 0.8 }],
        'categories:accessibility': ['error', { minScore: 0.9 }],
        'categories:best-practices': ['warn', { minScore: 0.8 }],
        'categories:seo': ['warn', { minScore: 0.8 }],
      },
    },
    upload: {
      target: 'temporary-public-storage',
    },
  },
};
```

### 8. 部署脚本 / Deployment Scripts

#### 8.1 自动化部署 / Automated Deployment

**部署脚本**:

```bash
#!/bin/bash
# deploy.sh

set -e

# 配置变量
ENVIRONMENT=$1
VERSION=$2

if [ -z "$ENVIRONMENT" ] || [ -z "$VERSION" ]; then
    echo "Usage: ./deploy.sh <environment> <version>"
    echo "Example: ./deploy.sh staging v1.0.0"
    exit 1
fi

echo "Deploying version $VERSION to $ENVIRONMENT..."

# 构建 Docker 镜像
echo "Building Docker images..."
docker build -t formal-model/frontend:$VERSION ./frontend
docker build -t formal-model/backend:$VERSION ./backend

# 推送镜像到仓库
echo "Pushing images to registry..."
docker push formal-model/frontend:$VERSION
docker push formal-model/backend:$VERSION

# 更新 Kubernetes 部署
echo "Updating Kubernetes deployment..."
kubectl set image deployment/formal-model-frontend frontend=formal-model/frontend:$VERSION -n $ENVIRONMENT
kubectl set image deployment/formal-model-backend backend=formal-model/backend:$VERSION -n $ENVIRONMENT

# 等待部署完成
echo "Waiting for deployment to complete..."
kubectl rollout status deployment/formal-model-frontend -n $ENVIRONMENT
kubectl rollout status deployment/formal-model-backend -n $ENVIRONMENT

# 运行健康检查
echo "Running health checks..."
./scripts/health-check.sh $ENVIRONMENT

echo "Deployment completed successfully!"
```

**健康检查脚本**:

```bash
#!/bin/bash
# scripts/health-check.sh

ENVIRONMENT=$1
BASE_URL="https://api.$ENVIRONMENT.formalmodel.com"

echo "Running health checks for $ENVIRONMENT..."

# 检查 API 健康状态
echo "Checking API health..."
response=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/health")
if [ $response -eq 200 ]; then
    echo "✅ API health check passed"
else
    echo "❌ API health check failed: $response"
    exit 1
fi

# 检查数据库连接
echo "Checking database connection..."
response=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/health/db")
if [ $response -eq 200 ]; then
    echo "✅ Database health check passed"
else
    echo "❌ Database health check failed: $response"
    exit 1
fi

# 检查缓存连接
echo "Checking cache connection..."
response=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/health/cache")
if [ $response -eq 200 ]; then
    echo "✅ Cache health check passed"
else
    echo "❌ Cache health check failed: $response"
    exit 1
fi

echo "All health checks passed! 🎉"
```

### 9. 开发工具集成 / Development Tools Integration

#### 9.1 IDE 插件推荐 / IDE Plugin Recommendations

**VS Code 插件**:

```json
// .vscode/extensions.json
{
  "recommendations": [
    "ms-vscode.vscode-typescript-next",
    "ms-python.python",
    "ms-python.black-formatter",
    "ms-python.pylint",
    "bradlc.vscode-tailwindcss",
    "esbenp.prettier-vscode",
    "ms-vscode.vscode-json",
    "redhat.vscode-yaml",
    "ms-azuretools.vscode-docker",
    "ms-kubernetes-tools.vscode-kubernetes-tools",
    "ms-vscode.vscode-jest",
    "ms-vscode.vscode-js-debug",
    "ms-vscode.vscode-js-debug-companion"
  ]
}
```

**IntelliJ IDEA 插件**:

- Python
- Docker
- Kubernetes
- Git Integration
- Database Tools
- REST Client
- Rainbow Brackets
- Material Theme UI

#### 9.2 命令行工具 / Command Line Tools

**开发工具脚本**:

```bash
#!/bin/bash
# scripts/dev-tools.sh

# 开发环境快速启动
function start_dev() {
    echo "Starting development environment..."
    docker-compose up -d postgres redis
    cd frontend && npm run dev &
    cd ../backend && python app.py &
    echo "Development environment started!"
}

# 停止开发环境
function stop_dev() {
    echo "Stopping development environment..."
    docker-compose down
    pkill -f "npm run dev"
    pkill -f "python app.py"
    echo "Development environment stopped!"
}

# 代码格式化
function format_code() {
    echo "Formatting code..."
    cd frontend && npm run format
    cd ../backend && black . && isort .
    echo "Code formatting completed!"
}

# 运行测试
function run_tests() {
    echo "Running tests..."
    cd frontend && npm test
    cd ../backend && python -m pytest
    echo "Tests completed!"
}

# 代码检查
function lint_code() {
    echo "Running linting..."
    cd frontend && npm run lint
    cd ../backend && pylint app/
    echo "Linting completed!"
}

# 构建项目
function build_project() {
    echo "Building project..."
    cd frontend && npm run build
    cd ../backend && python setup.py build
    echo "Build completed!"
}

# 显示帮助信息
function show_help() {
    echo "Available commands:"
    echo "  start_dev     - Start development environment"
    echo "  stop_dev      - Stop development environment"
    echo "  format_code   - Format code"
    echo "  run_tests     - Run tests"
    echo "  lint_code     - Run linting"
    echo "  build_project - Build project"
    echo "  help          - Show this help"
}

# 主函数
case "$1" in
    start_dev)
        start_dev
        ;;
    stop_dev)
        stop_dev
        ;;
    format_code)
        format_code
        ;;
    run_tests)
        run_tests
        ;;
    lint_code)
        lint_code
        ;;
    build_project)
        build_project
        ;;
    help|*)
        show_help
        ;;
esac
```

## 总结 / Summary

开发工具和环境设置指南为形式化模型平台提供了完整的开发环境配置方案。通过标准化的工具链和自动化流程，确保开发团队能够高效、高质量地完成项目开发。

工具配置的成功实施将为项目的技术实施提供强有力的支撑，确保代码质量、测试覆盖率和部署效率达到预期目标。

---

*最后更新: 2025-08-01*  
*版本: 1.0.0*
