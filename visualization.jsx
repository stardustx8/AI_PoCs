import React, { useState, useEffect } from 'react';
import { ArrowRight, Database, Search, Bot, FileText, Brain, MessageSquare, Code } from 'lucide-react';

const RAGFlowVisualization = () => {
  const [activeStage, setActiveStage] = useState(null);
  
  // Define all stages and their properties
  const stages = {
    upload: { 
      title: 'Document Upload', 
      color: 'bg-blue-500',
      icon: FileText,
      description: 'Raw document is uploaded and prepared for processing',
      detailText: 'The document is read and validated'
    },
    chunk: { 
      title: 'Text Chunking', 
      color: 'bg-green-500',
      icon: Code,
      description: 'Document is split into smaller, manageable pieces',
      detailText: 'Text is divided into semantic chunks using paragraph breaks'
    },
    embed: { 
      title: 'Vector Embedding', 
      color: 'bg-yellow-500',
      icon: Brain,
      description: 'Each chunk is converted into a numerical vector',
      detailText: 'OpenAI embeddings model converts text to high-dimensional vectors'
    },
    store: { 
      title: 'Vector Storage', 
      color: 'bg-purple-500',
      icon: Database,
      description: 'Vectors are stored in the database',
      detailText: 'ChromaDB stores vectors with metadata for quick retrieval'
    },
    query: { 
      title: 'Query Processing', 
      color: 'bg-pink-500',
      icon: MessageSquare,
      description: 'User question is processed',
      detailText: 'Question is converted to a vector for similarity search'
    },
    retrieve: { 
      title: 'Context Retrieval', 
      color: 'bg-orange-500',
      icon: Search,
      description: 'Most relevant passages are found',
      detailText: 'Similar vectors are retrieved from the database'
    },
    generate: { 
      title: 'Answer Generation', 
      color: 'bg-red-500',
      icon: Bot,
      description: 'Final answer is generated',
      detailText: 'GPT model generates answer using retrieved context'
    }
  };

  // Component for each stage in the pipeline
  const StageBox = ({ stage }) => {
    const stageInfo = stages[stage];
    const Icon = stageInfo.icon;
    const isActive = activeStage === stage;
    
    return (
      <div 
        className={`relative p-4 rounded-lg border-2 transition-all duration-300
          ${isActive ? `${stageInfo.color} text-white shadow-lg scale-105` : 'bg-white border-gray-200'}
          hover:shadow-md cursor-pointer`}
        onMouseEnter={() => setActiveStage(stage)}
        onMouseLeave={() => setActiveStage(null)}
      >
        <div className="flex items-center space-x-3">
          <Icon className={`w-6 h-6 ${isActive ? 'text-white' : 'text-gray-600'}`} />
          <span className="font-medium">{stageInfo.title}</span>
        </div>
        {isActive && (
          <div className="absolute top-full left-0 mt-2 w-64 p-3 bg-gray-800 text-white text-sm rounded-md z-10 shadow-lg">
            <p className="font-medium mb-2">{stageInfo.description}</p>
            <p className="text-gray-300 text-xs">{stageInfo.detailText}</p>
          </div>
        )}
      </div>
    );
  };

  // Arrow component for showing flow direction
  const Arrow = () => (
    <div className="flex-shrink-0">
      <ArrowRight className="w-6 h-6 text-gray-400" />
    </div>
  );

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gray-50 rounded-xl shadow-sm">
      <h2 className="text-2xl font-bold mb-8 text-center text-gray-800">RAG Pipeline Visualization</h2>
      
      {/* Document Processing Flow */}
      <div className="mb-12">
        <h3 className="text-lg font-semibold mb-4 text-gray-700">Document Processing Pipeline</h3>
        <div className="flex items-center space-x-4">
          <StageBox stage="upload" />
          <Arrow />
          <StageBox stage="chunk" />
          <Arrow />
          <StageBox stage="embed" />
          <Arrow />
          <StageBox stage="store" />
        </div>
      </div>

      {/* Query Processing Flow */}
      <div>
        <h3 className="text-lg font-semibold mb-4 text-gray-700">Query Processing Pipeline</h3>
        <div className="flex items-center space-x-4">
          <StageBox stage="query" />
          <Arrow />
          <StageBox stage="retrieve" />
          <Arrow />
          <StageBox stage="generate" />
        </div>
      </div>
    </div>
  );
};

// Render the component
const root = document.getElementById('rag-root');
if (root) {
  ReactDOM.render(React.createElement(RAGFlowVisualization), root);
}

export default RAGFlowVisualization;