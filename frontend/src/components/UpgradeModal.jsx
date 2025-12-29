import { useState } from 'react';
import { CreditCard } from 'lucide-react';

const UpgradeModal = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white p-6 rounded-lg w-96">
        <h2 className="text-2xl font-bold mb-4 flex items-center">
          <CreditCard className="w-6 h-6 mr-2" />
          Upgrade to Premium
        </h2>
        <p className="mb-4">Unlimited conversions, no limits!</p>
        <ul className="mb-4 space-y-2">
          <li>• Unlimited daily uses</li>
          <li>• Priority processing</li>
          <li>• $9.99/month</li>
        </ul>
        <button className="w-full bg-purple-600 text-white py-2 rounded mb-2 hover:bg-purple-700">
          Subscribe with Stripe (Stub)
        </button>
        <button onClick={onClose} className="w-full text-gray-500">Cancel</button>
      </div>
    </div>
  );
};

export default UpgradeModal;