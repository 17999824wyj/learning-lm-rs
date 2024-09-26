import type { AxiosProgressEvent, GenericAbortSignal } from 'axios'
import { post } from '@/utils/request'
import { useAuthStore, useSettingStore } from '@/store'

export function fetchChatAPI<T = any>(
  prompt: string,
  options?: { conversationId?: string; parentMessageId?: string },
  signal?: GenericAbortSignal,
) {
  return post<T>({
    url: '/infer',
    data: { prompt, options },
    signal,
  })
}

export function fetchChatConfig<T = any>() {
  return post<T>({
    url: '/config',
  })
}

export function fetchChatAPIProcess<T = any>({
  prompt,
  options,
  signal,
  onDownloadProgress,
}: {
  prompt: string
  options?: { conversationId?: string; parentMessageId?: string }
  signal?: GenericAbortSignal
  onDownloadProgress?: (progressEvent: AxiosProgressEvent) => void
}) {
  const settingStore = useSettingStore()
  const authStore = useAuthStore()

  let data: Record<string, any> = {
    inputs: prompt,
  }

  if (options) {
    // 添加 options 中不为空的属性到 data.options
    if (options.conversationId)
      data.options = { ...data.options, conversationId: options.conversationId }

    if (options.parentMessageId)
      data.options = { ...data.options, parentMessageId: options.parentMessageId }
  }

  if (authStore.isInfiniLM) {
    data = {
      ...data,
      temperature: settingStore.temperature,
      top_p: settingStore.top_p,
      top_k: settingStore.top_k,
    }
  }

  return post<T>({
    url: '/infer',
    data,
    signal,
    onDownloadProgress,
  })
}

export function fetchSession<T>() {
  // Not finished
  return post<T>({
    url: '/404',
  })
}

export function fetchVerify<T>(token: string) {
  // Not finished
  return post<T>({
    url: '/verify',
    data: { token },
  })
}
