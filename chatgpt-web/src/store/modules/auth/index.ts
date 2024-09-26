import { defineStore } from 'pinia'
import { getToken, removeToken, setToken } from './helper'
import { store } from '@/store/helper'
// import { fetchSession } from '@/api'

interface SessionResponse {
  auth: boolean
  model: 'InfiniLM'
}

export interface AuthState {
  token: string | undefined
  session: SessionResponse | null
}

export const useAuthStore = defineStore('auth-store', {
  state: (): AuthState => ({
    token: getToken(),
    session: null,
  }),

  getters: {
    isInfiniLM(state): boolean {
      return state.session?.model === 'InfiniLM'
    },
  },

  actions: {
    async getSession() {
      try {
        const data = {
          auth: true,
          model: 'InfiniLM',
        }
        // const data = fetchSession<SessionResponse>();
        this.session = { ...data }
        return Promise.resolve(data)
      }
      catch (error) {
        return Promise.reject(error)
      }
    },

    setToken(token: string) {
      this.token = token
      setToken(token)
    },

    removeToken() {
      this.token = undefined
      removeToken()
    },
  },
})

export function useAuthStoreWithout() {
  return useAuthStore(store)
}
