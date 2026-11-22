import { useCallback, useEffect, useRef, useState } from 'react'
import { FileText, MessageSquareText, Plus, Search, Send, Sparkles, Trash2, UploadCloud, X, Menu, PanelLeftClose, BookOpen, ShieldCheck, CircleHelp } from 'lucide-react'

const API = (import.meta.env.VITE_API_URL || 'http://localhost:8000/api').replace(/\/$/, '')
const suggestions = ['What are the key takeaways?', 'Summarize the main argument', 'Compare the recommendations']

async function request(path, options = {}) {
  const response = await fetch(`${API}${path}`, options)
  if (!response.ok) {
    let message = `Request failed (${response.status})`
    try { message = (await response.json()).detail || message } catch { /* response was not JSON */ }
    throw new Error(message)
  }
  return response.json()
}

function App() {
  const [documents, setDocuments] = useState([])
  const [messages, setMessages] = useState([])
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')
  const [health, setHealth] = useState(null)
  const [view, setView] = useState('chat')
  const [mobileOpen, setMobileOpen] = useState(false)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [sessionId, setSessionId] = useState(() => crypto.randomUUID())
  const [history, setHistory] = useState([])
  const [activeHistory, setActiveHistory] = useState('')
  const [search, setSearch] = useState('')
  const fileInput = useRef(null)
  const bottom = useRef(null)

  const refresh = useCallback(async () => {
    try {
      const [docs, status] = await Promise.all([request('/documents'), request('/health')])
      setDocuments(docs.documents)
      setHealth(status)
      setError('')
    } catch (err) { setError(err.message) }
  }, [])

  useEffect(() => { refresh() }, [refresh])
  useEffect(() => { bottom.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, loading])

  async function uploadFiles(files) {
    if (!files?.length) return
    setUploading(true); setError('')
    const data = new FormData()
    Array.from(files).forEach(file => data.append('files', file))
    try {
      const result = await request('/upload', { method: 'POST', body: data })
      const notices = result.uploaded_files.filter(file => file.status !== 'success' || file.message?.startsWith('No selectable text'))
      if (notices.length) setError(notices.map(file => `${file.filename}: ${file.message}`).join(' · '))
      await refresh()
      setView('documents')
    } catch (err) { setError(err.message) }
    finally { setUploading(false); if (fileInput.current) fileInput.current.value = '' }
  }

  async function deleteDocument(id) {
    try { await request(`/documents/${id}`, { method: 'DELETE' }); await refresh() }
    catch (err) { setError(err.message) }
  }

  async function sendMessage(text = query) {
    const clean = text.trim()
    if (!clean || loading) return
    const userMessage = { role: 'user', content: clean }
    setMessages(current => [...current, userMessage]); setQuery(''); setLoading(true); setError('')
    try {
      const answer = await request('/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query: clean, session_id: sessionId, top_k: 4 }) })
      setMessages(current => [...current, { role: 'assistant', content: answer.answer, citations: answer.citations, found: answer.found_relevant_context }])
      setHistory(current => {
        const title = clean.length > 42 ? `${clean.slice(0, 42)}…` : clean
        const next = [{ id: sessionId, title }, ...current.filter(item => item.id !== sessionId)]
        localStorage.setItem('smartdocs-history', JSON.stringify(next))
        return next
      })
    } catch (err) { setError(err.message) }
    finally { setLoading(false) }
  }

  useEffect(() => { try { setHistory(JSON.parse(localStorage.getItem('smartdocs-history') || '[]')) } catch { setHistory([]) } }, [])
  useEffect(() => {
    if (!messages.length) return
    try {
      const sessions = JSON.parse(localStorage.getItem('smartdocs-sessions') || '{}')
      sessions[sessionId] = messages
      localStorage.setItem('smartdocs-sessions', JSON.stringify(sessions))
    } catch { /* Keep the current conversation usable when browser storage is unavailable. */ }
  }, [messages, sessionId])

  function newChat() {
    if (messages.length) {
      const title = messages.find(item => item.role === 'user')?.content.slice(0, 42) || 'New conversation'
      setHistory(current => {
        const next = [{ id: sessionId, title }, ...current.filter(item => item.id !== sessionId)]
        localStorage.setItem('smartdocs-history', JSON.stringify(next))
        return next
      })
    }
    setMessages([]); setSessionId(crypto.randomUUID()); setActiveHistory(''); setView('chat'); setMobileOpen(false)
  }

  const nav = (
    <>
      <div className="brand"><div className="brand-mark"><BookOpen size={19} /></div><span>smartdocs</span><button className="icon-button collapse" aria-label="Close menu" onClick={() => setMobileOpen(false)}><X size={18} /></button></div>
      <button className={`nav-item ${view === 'chat' ? 'active' : ''}`} onClick={newChat}><MessageSquareText size={18} />New conversation</button>
      <button className={`nav-item ${view === 'documents' ? 'active' : ''}`} onClick={() => { setView('documents'); setMobileOpen(false) }}><FileText size={18} />My documents <span className="count">{documents.length}</span></button>
      <div className="side-heading">YOUR LIBRARY</div>
      <div className="side-list">
        {history.length ? history.map(item => <button key={item.id} className={`history-item ${activeHistory === item.id ? 'selected' : ''}`} onClick={() => { let saved = []; try { saved = JSON.parse(localStorage.getItem('smartdocs-sessions') || '{}')[item.id] || [] } catch { /* Ignore unavailable browser storage. */ } setActiveHistory(item.id); setMessages(saved); setSessionId(item.id); setView('chat'); setMobileOpen(false) }}><MessageSquareText size={15} /><span>{item.title}</span></button>) : <div className="empty-side">Your conversations will appear here.</div>}
      </div>
      <div className="side-bottom">
        <div className="privacy"><ShieldCheck size={17} /><span>Your files stay in your workspace.</span></div>
        <div className="profile"><div className="avatar">S</div><div><strong>Workspace</strong><small>Personal library</small></div><CircleHelp size={17} className="help" /></div>
      </div>
    </>
  )

  return <div className={`app-shell ${sidebarCollapsed ? 'sidebar-collapsed' : ''}`}>
    <aside className={`sidebar ${mobileOpen ? 'mobile-open' : ''}`}>{nav}</aside>
    {mobileOpen && <button className="scrim" aria-label="Close navigation" onClick={() => setMobileOpen(false)} />}
    <main className="main-area">
      <header className="topbar"><button className="icon-button mobile-menu" aria-label="Open navigation" onClick={() => setMobileOpen(true)}><Menu size={20} /></button><div className="crumb">Workspace <span>/</span> <b>{view === 'chat' ? 'Research assistant' : 'My documents'}</b></div><div className="top-right"><span className={`status-dot ${health ? 'online' : ''}`} />{health ? 'API connected' : 'Connecting…'}<button className="icon-button desktop-collapse" aria-label="Toggle sidebar" onClick={() => setSidebarCollapsed(value => !value)}><PanelLeftClose size={18} /></button></div></header>
      {error && <div className="toast" role="alert"><span>{error}</span><button onClick={() => setError('')} aria-label="Dismiss"><X size={16} /></button></div>}
      {view === 'chat' ? <section className="chat-view">
        <div className={`conversation ${messages.length ? 'has-messages' : ''}`}>
          {!messages.length ? <div className="welcome">
            <div className="welcome-icon"><Sparkles size={22} /></div>
            <div className="eyebrow">AI DOCUMENT RESEARCH</div>
            <h1>Answers grounded<br />in <span>your documents.</span></h1>
            <p>Ask a question across your PDFs. SmartDocs will find the relevant passages and show you exactly where each answer comes from.</p>
            <div className="suggestion-label">TRY ASKING</div>
            <div className="suggestions">{suggestions.map(item => <button key={item} onClick={() => sendMessage(item)}>{item}<span>↗</span></button>)}</div>
            {!documents.length && <button className="inline-upload" onClick={() => fileInput.current?.click()}><UploadCloud size={17} />Upload your first PDF</button>}
          </div> : <div className="message-list">{messages.map((message, index) => <article key={index} className={`message ${message.role}`}>
            <div className="message-avatar">{message.role === 'user' ? 'Y' : <Sparkles size={15} />}</div>
            <div className="message-body"><div className="message-label">{message.role === 'user' ? 'You' : 'SmartDocs'}</div><div className="message-content">{message.content.split(/\n/).map((line, i) => <p key={i}>{line || <br />}</p>)}</div>
              {message.citations?.length > 0 && <div className="sources"><div className="sources-title"><span><BookOpen size={15} />Sources</span><small>{message.citations.length} passages</small></div><div className="source-grid">{message.citations.map(source => <details className="source-card" key={source.citation_index}><summary><span className="source-number">{source.citation_index}</span><span className="source-meta"><b>{source.filename}</b><small>Page {source.page_number} · {Math.round(source.relevance_score * 100)}% match</small></span><span className="chevron">⌄</span></summary><blockquote>{source.snippet}</blockquote></details>)}</div></div>}
            </div>
          </article>)}{loading && <article className="message assistant"><div className="message-avatar"><Sparkles size={15} /></div><div className="message-body"><div className="message-label">SmartDocs</div><div className="typing"><i /><i /><i /><span>Searching your documents…</span></div></div></article>}<div ref={bottom} /></div>}
        </div>
        <div className="composer-wrap"><div className="composer"><textarea aria-label="Ask about your documents" value={query} onChange={event => setQuery(event.target.value)} onKeyDown={event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); sendMessage() } }} placeholder="Ask a question about your documents…" rows="1" /><button className="send-button" disabled={!query.trim() || loading} aria-label="Send question" onClick={() => sendMessage()}><Send size={17} /></button></div><div className="composer-note">AI can make mistakes. Verify important details in the cited sources.</div></div>
      </section> : <section className="documents-view">
        <div className="page-heading"><div><div className="eyebrow">YOUR KNOWLEDGE BASE</div><h1>My documents</h1><p>Manage the PDFs SmartDocs can search and reference.</p></div><button className="primary-button" onClick={() => fileInput.current?.click()} disabled={uploading}><Plus size={17} />{uploading ? 'Uploading…' : 'Upload PDFs'}</button></div>
        <div className="doc-toolbar"><div className="search-box"><Search size={17} /><input value={search} onChange={event => setSearch(event.target.value)} placeholder="Search documents" /></div><span>{documents.length} {documents.length === 1 ? 'document' : 'documents'}</span></div>
        {documents.filter(doc => doc.filename.toLowerCase().includes(search.toLowerCase())).length ? <div className="doc-table"><div className="doc-row table-head"><span>DOCUMENT</span><span>PAGES</span><span>CHUNKS</span><span>ADDED</span><span /></div>{documents.filter(doc => doc.filename.toLowerCase().includes(search.toLowerCase())).map(doc => <div className="doc-row" key={doc.document_id}><div className="doc-name"><div className="pdf-icon"><FileText size={18} /></div><div><b>{doc.filename}</b><small>{(doc.file_size_bytes / 1024 / 1024).toFixed(2)} MB</small></div></div><span>{doc.total_pages}</span><span>{doc.chunk_count}</span><span>{new Date(doc.upload_date).toLocaleDateString()}</span><button className="icon-button delete-button" aria-label={`Delete ${doc.filename}`} onClick={() => deleteDocument(doc.document_id)}><Trash2 size={17} /></button></div>)}</div> : <div className="empty-docs"><div className="empty-icon"><FileText size={25} /></div><h3>{search ? 'No matching documents' : 'Your library is empty'}</h3><p>{search ? 'Try a different name.' : 'Upload PDFs to start asking questions across your research.'}</p>{!search && <button className="primary-button" onClick={() => fileInput.current?.click()}><UploadCloud size={17} />Choose PDF files</button>}</div>}
      </section>}
    </main>
    <input ref={fileInput} type="file" accept="application/pdf,.pdf" multiple hidden onChange={event => uploadFiles(event.target.files)} />
  </div>
}

export default App
