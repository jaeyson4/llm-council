import { useState } from 'react';
import './Sidebar.css';

export default function Sidebar({
  conversations,
  currentConversationId,
  onSelectConversation,
  onNewConversation,
  onDeleteConversation,
}) {
  // Which conversation is awaiting delete confirmation (null = none).
  const [confirmingId, setConfirmingId] = useState(null);

  const handleDeleteClick = (e, id) => {
    e.stopPropagation(); // don't also select the row
    setConfirmingId(id); // reveal the confirm / cancel controls
  };

  const handleConfirmDelete = (e, id) => {
    e.stopPropagation();
    setConfirmingId(null);
    onDeleteConversation(id);
  };

  const handleCancelDelete = (e) => {
    e.stopPropagation();
    setConfirmingId(null);
  };

  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <h1>LLM Council</h1>
        <button className="new-conversation-btn" onClick={onNewConversation}>
          + New Conversation
        </button>
      </div>

      <div className="conversation-list">
        {conversations.length === 0 ? (
          <div className="no-conversations">No conversations yet</div>
        ) : (
          conversations.map((conv) => (
            <div
              key={conv.id}
              className={`conversation-item ${
                conv.id === currentConversationId ? 'active' : ''
              }`}
              onClick={() => onSelectConversation(conv.id)}
            >
              <div className="conversation-body">
                <div className="conversation-title">
                  {conv.title || 'New Conversation'}
                </div>
                <div className="conversation-meta">
                  {conv.message_count} messages
                </div>
              </div>

              {confirmingId === conv.id ? (
                <div
                  className="conversation-confirm"
                  onClick={(e) => e.stopPropagation()}
                >
                  <span className="confirm-label">Delete?</span>
                  <button
                    className="confirm-delete-btn"
                    title="Confirm delete"
                    onClick={(e) => handleConfirmDelete(e, conv.id)}
                  >
                    Yes
                  </button>
                  <button
                    className="cancel-delete-btn"
                    title="Cancel"
                    onClick={handleCancelDelete}
                  >
                    No
                  </button>
                </div>
              ) : (
                <button
                  className="delete-conversation-btn"
                  title="Delete conversation"
                  aria-label="Delete conversation"
                  onClick={(e) => handleDeleteClick(e, conv.id)}
                >
                  ×
                </button>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
