import React, { useState } from 'react';
import {
  Box,
  Button,
  Input,
  useToast,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalFooter,
  ModalBody,
  ModalCloseButton,
  useDisclosure,
  Text,
  VStack,
  Heading,
  Container,
  FormControl,
  FormLabel,
  FormErrorMessage,
} from '@chakra-ui/react';

const SentimentAnalysisComponent = () => {
  const [inputText, setInputText] = useState('');
  const [isInputInvalid, setIsInputInvalid] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState('');
  const { isOpen, onOpen, onClose } = useDisclosure();
  const toast = useToast();

  const validateInput = () => {
    if (!inputText.trim()) {
      setIsInputInvalid(true);
      toast({
        title: 'Validation Error',
        description: "Please enter some text to analyze.",
        status: 'warning',
        duration: 5000,
        isClosable: true,
      });
      return false;
    }
    setIsInputInvalid(false);
    return true;
  };

  const handleSubmit = async () => {
    if (!validateInput()) return;

    setIsLoading(true);
    try {
      const response = await fetch('YOUR_API_ENDPOINT', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: inputText }),
      });
      const data = await response.json();

      setResult(data.result); // Assuming the API returns { result: 'positive' } or { result: 'negative' }
      onOpen();
    } catch (error) {
  
      toast({
        title: 'Network Error',
        description: 'Could not reach the analysis service. Please try again later.',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Container maxW="xl" centerContent>
      <Box p={5} shadow="md" borderWidth="1px" borderRadius="md" w="full" mt={10}>
        <Heading mb={6} textAlign="center">Sentiment Analysis Tool</Heading>
        <VStack spacing={4}>
          <FormControl isInvalid={isInputInvalid}>
            <FormLabel htmlFor='text'>Enter your text</FormLabel>
            <Input
              id='text'
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Type something..."
              size="lg"
            />
            <FormErrorMessage>Text is required for analysis.</FormErrorMessage>
          </FormControl>
          <Button
            colorScheme="teal"
            isLoading={isLoading}
            loadingText="Analyzing"
            onClick={handleSubmit}
            size="lg"
            w="full"
          >
            Analyze Sentiment
          </Button>
        </VStack>
      </Box>

      <Modal isOpen={isOpen} onClose={onClose} isCentered>
        <ModalOverlay />
        <ModalContent>
          <ModalHeader>Analysis Result</ModalHeader>
          <ModalCloseButton />
          <ModalBody pb={6}>
            <Text fontSize="xl" textAlign="center">
              The sentiment of the entered text is <strong>{result === 'positive' ? 'Positive 😊' : 'Negative 😞'}</strong>
            </Text>
          </ModalBody>
          <ModalFooter>
            <Button colorScheme="blue" mr={3} onClick={onClose}>
              Close
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </Container>
  );
};

export default SentimentAnalysisComponent;
